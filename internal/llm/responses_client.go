package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"
)

var (
	bearerTokenRe         = regexp.MustCompile(`(?i)Bearer\s+[A-Za-z0-9._~+/=-]+`)
	sensitiveJSONFieldRe  = regexp.MustCompile(`(?i)("(?:authorization|access_token|api_key|ChatGPT-Account-ID)"\s*:\s*")[^"]*(")`)
	sensitivePlainFieldRe = regexp.MustCompile(`(?i)\b(authorization|access_token|api_key|ChatGPT-Account-ID)\b\s*[:=]\s*[^\s,}]+`)
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

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		respBody, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("responses API returned %s: %s", httpResp.Status, redactSensitiveString(string(respBody)))
	}

	if strings.Contains(strings.ToLower(httpResp.Header.Get("Content-Type")), "text/event-stream") {
		return mapResponsesStreamReader(httpResp.Body)
	}

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, err
	}
	if isEventStream("", respBody) {
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
		"stream":       strings.EqualFold(c.cfg.AuthMode, "chatgpt"),
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
			if len(msg.ResponsesOutput) > 0 {
				for _, raw := range msg.ResponsesOutput {
					var item map[string]any
					if err := json.Unmarshal(raw, &item); err != nil {
						return "", nil, fmt.Errorf("parse preserved responses output item: %w", err)
					}
					input = append(input, item)
				}
				continue
			}
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
	type responsesOutputItem struct {
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
	}
	var parsed struct {
		ID                string            `json:"id"`
		Model             string            `json:"model"`
		Status            string            `json:"status"`
		OutputText        string            `json:"output_text"`
		RawOutput         []json.RawMessage `json:"output"`
		IncompleteDetails any               `json:"incomplete_details"`
		Error             any               `json:"error"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parse responses response: %w", err)
	}
	if err := validateResponsesStatus(parsed.Status, parsed.Error, parsed.IncompleteDetails); err != nil {
		return nil, err
	}

	var textParts []string
	if parsed.OutputText != "" {
		textParts = append(textParts, parsed.OutputText)
	}
	var toolCalls []ToolCall
	for _, rawItem := range parsed.RawOutput {
		var item responsesOutputItem
		if err := json.Unmarshal(rawItem, &item); err != nil {
			return nil, fmt.Errorf("parse responses output item: %w", err)
		}
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
		ID:              parsed.ID,
		Model:           parsed.Model,
		ResponsesOutput: cloneRawMessages(parsed.RawOutput),
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
	return mapResponsesStreamReader(bytes.NewReader(raw))
}

func mapResponsesStreamReader(r io.Reader) (*ChatResponse, error) {
	streamDone := fmt.Errorf("responses stream done")
	var finalRaw []byte
	var textParts []string
	toolCallsByID := make(map[string]*ToolCall)
	itemIDToCallID := make(map[string]string)
	var toolCallOrder []string

	addToolCallID := func(id string) {
		if id == "" {
			return
		}
		if _, ok := toolCallsByID[id]; !ok {
			toolCallsByID[id] = &ToolCall{ID: id, Type: "function"}
			toolCallOrder = append(toolCallOrder, id)
		}
	}
	replaceOrderedToolCallID := func(oldID, newID string) {
		for i, orderedID := range toolCallOrder {
			if orderedID == oldID {
				toolCallOrder[i] = newID
				return
			}
		}
		toolCallOrder = append(toolCallOrder, newID)
	}
	removeOrderedToolCallID := func(id string) {
		for i, orderedID := range toolCallOrder {
			if orderedID == id {
				toolCallOrder = append(toolCallOrder[:i], toolCallOrder[i+1:]...)
				return
			}
		}
	}
	canonicalToolCallID := func(itemID, callID string) string {
		if itemID != "" && callID != "" {
			itemIDToCallID[itemID] = callID
			if itemTC, ok := toolCallsByID[itemID]; ok {
				canonicalTC, exists := toolCallsByID[callID]
				if !exists {
					itemTC.ID = callID
					toolCallsByID[callID] = itemTC
					replaceOrderedToolCallID(itemID, callID)
					delete(toolCallsByID, itemID)
					return callID
				}
				if canonicalTC.Function.Name == "" {
					canonicalTC.Function.Name = itemTC.Function.Name
				}
				if canonicalTC.Function.Arguments == "" {
					canonicalTC.Function.Arguments = itemTC.Function.Arguments
				}
				delete(toolCallsByID, itemID)
				removeOrderedToolCallID(itemID)
				return callID
			}
			addToolCallID(callID)
			return callID
		}
		if callID != "" {
			return callID
		}
		if mapped := itemIDToCallID[itemID]; mapped != "" {
			return mapped
		}
		return itemID
	}
	getToolCall := func(itemID, callID string) *ToolCall {
		id := canonicalToolCallID(itemID, callID)
		if id == "" {
			return nil
		}
		addToolCallID(id)
		return toolCallsByID[id]
	}

	processEvent := func(eventName string, dataLines []string) error {
		if len(dataLines) == 0 {
			return nil
		}
		data := strings.Join(dataLines, "\n")
		if strings.TrimSpace(data) == "[DONE]" {
			return streamDone
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
			return streamDone
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
			itemID, _ := item["id"].(string)
			tc := getToolCall(itemID, callID)
			if tc == nil {
				return nil
			}
			if name, _ := item["name"].(string); name != "" {
				tc.Function.Name = name
			}
			if args, _ := item["arguments"].(string); args != "" {
				tc.Function.Arguments = args
			}
		case "response.function_call_arguments.delta":
			callID, _ := ev["call_id"].(string)
			itemID, _ := ev["item_id"].(string)
			tc := getToolCall(itemID, callID)
			if tc == nil {
				return nil
			}
			if delta, _ := ev["delta"].(string); delta != "" {
				tc.Function.Arguments += delta
			}
		case "response.failed", "response.incomplete":
			if errObj, ok := ev["error"]; ok {
				b, _ := json.Marshal(errObj)
				return fmt.Errorf("responses stream failed: %s", redactSensitiveString(string(b)))
			}
			return fmt.Errorf("responses stream failed: %s", eventType)
		}
		return nil
	}

	finish := func() (*ChatResponse, error) {
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
				for _, id := range toolCallOrder {
					tc := toolCallsByID[id]
					if tc == nil {
						continue
					}
					if strings.TrimSpace(tc.Function.Name) == "" {
						continue
					}
					resp.Choices[0].Message.ToolCalls = append(resp.Choices[0].Message.ToolCalls, *tc)
				}
			}
			return resp, nil
		}

		var toolCalls []ToolCall
		for _, id := range toolCallOrder {
			tc := toolCallsByID[id]
			if tc == nil {
				continue
			}
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

	scanner := bufio.NewScanner(r)
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
				if err == streamDone {
					return finish()
				}
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
		if err == streamDone {
			return finish()
		}
		return nil, err
	}

	return finish()
}

func validateResponsesStatus(status string, errObj any, incompleteDetails any) error {
	status = strings.ToLower(strings.TrimSpace(status))
	if errObj != nil {
		return fmt.Errorf("responses API failed: %s", formatRedactedJSONValue(errObj))
	}
	switch status {
	case "", "completed":
		return nil
	case "failed", "cancelled":
		return fmt.Errorf("responses API failed with status %q", status)
	case "incomplete":
		return fmt.Errorf("responses API incomplete: %s", formatRedactedJSONValue(incompleteDetails))
	default:
		return fmt.Errorf("responses API unexpected status %q", status)
	}
}

func formatRedactedJSONValue(v any) string {
	if v == nil {
		return "null"
	}
	b, err := json.Marshal(v)
	if err != nil {
		return redactSensitiveString(fmt.Sprint(v))
	}
	return redactSensitiveString(string(b))
}

func redactSensitiveString(s string) string {
	s = bearerTokenRe.ReplaceAllString(s, "Bearer [REDACTED]")
	s = sensitiveJSONFieldRe.ReplaceAllString(s, `${1}[REDACTED]${2}`)
	s = sensitivePlainFieldRe.ReplaceAllString(s, `${1}=[REDACTED]`)
	return s
}

func cloneRawMessages(in []json.RawMessage) []json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	out := make([]json.RawMessage, len(in))
	for i := range in {
		out[i] = append(json.RawMessage(nil), in[i]...)
	}
	return out
}
