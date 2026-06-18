package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// ResponsesClient sends requests to the OpenAI Responses API.
type ResponsesClient struct {
	cfg        ClientConfig
	httpClient *http.Client
}

// NewResponsesClient creates a new Responses API client.
func NewResponsesClient(cfg ClientConfig) *ResponsesClient {
	if cfg.Timeout <= 0 {
		cfg.Timeout = 5 * time.Minute
	}
	cfg.URL = ensureResponsesSuffix(cfg.URL)
	return &ResponsesClient{
		cfg: cfg,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// CompletionsWithCtx sends a completion request to the Responses API.
func (c *ResponsesClient) CompletionsWithCtx(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = c.cfg.Model
	}
	body, err := c.buildResponsesBody(model, req)
	if err != nil {
		return nil, err
	}
	for k, v := range c.cfg.ExtraBody {
		body[k] = v
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal responses request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.URL, bytes.NewReader(raw))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.cfg.APIKey)
	httpReq.Header.Set("User-Agent", userAgent("responses"))
	if strings.EqualFold(c.cfg.AuthMode, "chatgpt") && c.cfg.AccountID != "" {
		httpReq.Header.Set("ChatGPT-Account-ID", c.cfg.AccountID)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, err
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, fmt.Errorf("responses API returned %s: %s", httpResp.Status, string(respBody))
	}

	if isEventStream(httpResp.Header.Get("Content-Type"), respBody) {
		return mapResponsesStream(respBody)
	}
	return mapResponsesResponse(respBody)
}

func (c *ResponsesClient) buildResponsesBody(model string, req ChatRequest) (map[string]any, error) {
	instructions, input, err := buildResponsesInstructionsAndInput(req.Messages)
	if err != nil {
		return nil, err
	}
	body := map[string]any{
		"model":        model,
		"instructions": instructions,
		"input":        input,
		"store":        false,
		"stream":       true,
	}
	if req.MaxTokens > 0 && !strings.EqualFold(c.cfg.AuthMode, "chatgpt") {
		body["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if len(req.Tools) > 0 {
		body["tools"] = buildResponsesTools(req.Tools)
	}
	return body, nil
}

func buildResponsesInstructionsAndInput(messages []Message) (string, []map[string]any, error) {
	var instructions []string
	input := make([]map[string]any, 0, len(messages))
	for _, msg := range messages {
		content := msg.ExtractText()
		switch msg.Role {
		case "system", "developer":
			if content != "" {
				instructions = append(instructions, content)
			}
		case "tool":
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": msg.ToolCallID,
				"output":  content,
			})
		case "assistant":
			if content != "" {
				input = append(input, map[string]any{
					"role":    "assistant",
					"content": content,
				})
			}
			for _, tc := range msg.ToolCalls {
				if strings.TrimSpace(tc.Function.Name) == "" {
					return "", nil, fmt.Errorf("responses assistant tool call %q missing function name", tc.ID)
				}
				input = append(input, map[string]any{
					"type":      "function_call",
					"call_id":   tc.ID,
					"name":      tc.Function.Name,
					"arguments": tc.Function.Arguments,
				})
			}
		default:
			role := msg.Role
			if role == "" {
				role = "user"
			}
			input = append(input, map[string]any{
				"role":    role,
				"content": content,
			})
		}
	}
	if len(instructions) == 0 {
		instructions = append(instructions, "You are a concise code review assistant.")
	}
	return strings.Join(instructions, "\n\n"), input, nil
}

func buildResponsesTools(tools []ToolDef) []map[string]any {
	out := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, map[string]any{
			"type":        "function",
			"name":        t.Function.Name,
			"description": t.Function.Description,
			"parameters":  t.Function.Parameters,
		})
	}
	return out
}

func mapResponsesResponse(raw []byte) (*ChatResponse, error) {
	var parsed struct {
		ID         string `json:"id"`
		Model      string `json:"model"`
		OutputText string `json:"output_text"`
		Output     []struct {
			Type      string `json:"type"`
			ID        string `json:"id"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
			Role      string `json:"role"`
			Content   []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Error any `json:"error"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parse responses response: %w", err)
	}

	var textParts []string
	if parsed.OutputText != "" {
		textParts = append(textParts, parsed.OutputText)
	}
	var toolCalls []ToolCall
	for _, item := range parsed.Output {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				if content.Text != "" {
					textParts = append(textParts, content.Text)
				}
			}
		case "function_call":
			if strings.TrimSpace(item.Name) == "" {
				continue
			}
			id := item.CallID
			if id == "" {
				id = item.ID
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:   id,
				Type: "function",
				Function: FunctionCall{
					Name:      item.Name,
					Arguments: item.Arguments,
				},
			})
		}
	}

	var contentPtr *string
	if len(textParts) > 0 {
		content := strings.TrimSpace(strings.Join(textParts, "\n"))
		contentPtr = &content
	}

	return &ChatResponse{
		ID:    parsed.ID,
		Model: parsed.Model,
		Choices: []Choice{{
			Message: ResponseMessage{
				Role:      "assistant",
				Content:   contentPtr,
				ToolCalls: toolCalls,
			},
			FinishReason: "stop",
		}},
		Usage: resolveUsage(raw),
	}, nil
}

func isEventStream(contentType string, body []byte) bool {
	if strings.Contains(strings.ToLower(contentType), "text/event-stream") {
		return true
	}
	trimmed := bytes.TrimSpace(body)
	return bytes.HasPrefix(trimmed, []byte("event:")) || bytes.HasPrefix(trimmed, []byte("data:"))
}

func mapResponsesStream(raw []byte) (*ChatResponse, error) {
	var finalRaw []byte
	var textParts []string
	toolCallsByID := make(map[string]*ToolCall)

	processEvent := func(eventName string, dataLines []string) error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		if strings.TrimSpace(data) == "[DONE]" {
			return nil
		}
		var ev map[string]any
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return fmt.Errorf("parse responses stream event %q: %w", eventName, err)
		}
		eventType, _ := ev["type"].(string)
		if eventType == "" {
			eventType = eventName
		}

		switch eventType {
		case "response.completed":
			if response, ok := ev["response"]; ok {
				b, err := json.Marshal(response)
				if err != nil {
					return err
				}
				finalRaw = b
			}
		case "response.output_text.delta":
			if delta, ok := ev["delta"].(string); ok {
				textParts = append(textParts, delta)
			}
		case "response.output_item.added", "response.output_item.done":
			item, ok := ev["item"].(map[string]any)
			if !ok || item["type"] != "function_call" {
				return nil
			}
			callID, _ := item["call_id"].(string)
			if callID == "" {
				callID, _ = item["id"].(string)
			}
			if callID == "" {
				return nil
			}
			tc, ok := toolCallsByID[callID]
			if !ok {
				tc = &ToolCall{ID: callID, Type: "function"}
				toolCallsByID[callID] = tc
			}
			if name, _ := item["name"].(string); name != "" {
				tc.Function.Name = name
			}
			if args, _ := item["arguments"].(string); args != "" {
				tc.Function.Arguments = args
			}
		case "response.function_call_arguments.delta":
			callID, _ := ev["call_id"].(string)
			if callID == "" {
				callID, _ = ev["item_id"].(string)
			}
			if callID == "" {
				return nil
			}
			tc, ok := toolCallsByID[callID]
			if !ok {
				tc = &ToolCall{ID: callID, Type: "function"}
				toolCallsByID[callID] = tc
			}
			if delta, _ := ev["delta"].(string); delta != "" {
				tc.Function.Arguments += delta
			}
		case "response.failed", "response.incomplete":
			if errObj, ok := ev["error"]; ok {
				b, _ := json.Marshal(errObj)
				return fmt.Errorf("responses stream failed: %s", string(b))
			}
			return fmt.Errorf("responses stream failed: %s", eventType)
		}
		return nil
	}

	scanner := bufio.NewScanner(bytes.NewReader(raw))
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)
	var eventName string
	var dataLines []string
	flush := func() error {
		if eventName == "" && len(dataLines) == 0 {
			return nil
		}
		err := processEvent(eventName, dataLines)
		eventName = ""
		dataLines = nil
		return err
	}
	for scanner.Scan() {
		line := strings.TrimRight(scanner.Text(), "\r")
		if strings.TrimSpace(line) == "" {
			if err := flush(); err != nil {
				return nil, err
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if err := flush(); err != nil {
		return nil, err
	}

	if len(finalRaw) > 0 {
		resp, err := mapResponsesResponse(finalRaw)
		if err != nil {
			return nil, err
		}
		if resp.Content() == "" && len(textParts) > 0 {
			content := strings.TrimSpace(strings.Join(textParts, ""))
			resp.Choices[0].Message.Content = &content
		}
		if len(resp.ToolCalls()) == 0 && len(toolCallsByID) > 0 {
			for _, tc := range toolCallsByID {
				if strings.TrimSpace(tc.Function.Name) == "" {
					continue
				}
				resp.Choices[0].Message.ToolCalls = append(resp.Choices[0].Message.ToolCalls, *tc)
			}
		}
		return resp, nil
	}

	var toolCalls []ToolCall
	for _, tc := range toolCallsByID {
		if strings.TrimSpace(tc.Function.Name) == "" {
			continue
		}
		toolCalls = append(toolCalls, *tc)
	}
	var contentPtr *string
	if len(textParts) > 0 {
		content := strings.TrimSpace(strings.Join(textParts, ""))
		contentPtr = &content
	}
	return &ChatResponse{
		Choices: []Choice{{
			Message: ResponseMessage{
				Role:      "assistant",
				Content:   contentPtr,
				ToolCalls: toolCalls,
			},
			FinishReason: "stop",
		}},
	}, nil
}
