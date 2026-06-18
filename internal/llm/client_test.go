package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

func TestNewOpenAIClient_URLNormalization(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "base URL without trailing slash",
			inputURL: "https://api.example.com/v1",
			wantURL:  "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "base URL with trailing slash",
			inputURL: "https://api.example.com/v1/",
			wantURL:  "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "full URL already has chat/completions",
			inputURL: "https://api.example.com/v1/chat/completions",
			wantURL:  "https://api.example.com/v1/chat/completions",
		},
		{
			name:     "full URL with trailing slash",
			inputURL: "https://api.example.com/v1/chat/completions/",
			wantURL:  "https://api.example.com/v1/chat/completions/",
		},
		{
			name:     "bare host",
			inputURL: "https://api.example.com",
			wantURL:  "https://api.example.com/chat/completions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewOpenAIClient(ClientConfig{URL: tt.inputURL})
			if client.cfg.URL != tt.wantURL {
				t.Errorf("got URL %q, want %q", client.cfg.URL, tt.wantURL)
			}
		})
	}
}

func TestNewAnthropicClient_URLNormalization(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "bare host",
			inputURL: "https://api.anthropic.com",
			wantURL:  "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "bare host with trailing slash",
			inputURL: "https://api.anthropic.com/",
			wantURL:  "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "full URL already has /v1/messages",
			inputURL: "https://api.anthropic.com/v1/messages",
			wantURL:  "https://api.anthropic.com/v1/messages",
		},
		{
			name:     "full URL with trailing slash",
			inputURL: "https://api.anthropic.com/v1/messages/",
			wantURL:  "https://api.anthropic.com/v1/messages/",
		},
		{
			name:     "custom proxy base URL",
			inputURL: "https://proxy.example.com/anthropic",
			wantURL:  "https://proxy.example.com/anthropic/v1/messages",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewAnthropicClient(ClientConfig{URL: tt.inputURL})
			if client.cfg.URL != tt.wantURL {
				t.Errorf("got URL %q, want %q", client.cfg.URL, tt.wantURL)
			}
		})
	}
}

func TestNewResponsesClient_URLNormalization(t *testing.T) {
	tests := []struct {
		name     string
		inputURL string
		wantURL  string
	}{
		{
			name:     "base URL without trailing slash",
			inputURL: "https://api.example.com/v1",
			wantURL:  "https://api.example.com/v1/responses",
		},
		{
			name:     "base URL with trailing slash",
			inputURL: "https://api.example.com/v1/",
			wantURL:  "https://api.example.com/v1/responses",
		},
		{
			name:     "full URL already has responses",
			inputURL: "https://api.example.com/v1/responses",
			wantURL:  "https://api.example.com/v1/responses",
		},
		{
			name:     "codex backend URL",
			inputURL: "https://chatgpt.com/backend-api/codex",
			wantURL:  "https://chatgpt.com/backend-api/codex/responses",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewResponsesClient(ClientConfig{URL: tt.inputURL})
			if client.cfg.URL != tt.wantURL {
				t.Errorf("got URL %q, want %q", client.cfg.URL, tt.wantURL)
			}
		})
	}
}

func TestResponsesClient_RequestAndTextOutput(t *testing.T) {
	var gotPath, gotAuthorization, gotAccountID string
	var gotBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuthorization = r.Header.Get("Authorization")
		gotAccountID = r.Header.Get("ChatGPT-Account-ID")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"resp_test",
			"model":"gpt-5.5",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],
			"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}
		}`))
	}))
	defer server.Close()

	client := NewResponsesClient(ClientConfig{
		URL:       server.URL + "/v1",
		APIKey:    "chatgpt-token",
		Model:     "gpt-5.5",
		AuthMode:  "chatgpt",
		AccountID: "account-123",
	})
	resp, err := client.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "system", Content: "be terse"}, {Role: "user", Content: "ping"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if gotPath != "/v1/responses" {
		t.Errorf("path = %q, want %q", gotPath, "/v1/responses")
	}
	if gotAuthorization != "Bearer chatgpt-token" {
		t.Errorf("Authorization = %q, want bearer token", gotAuthorization)
	}
	if gotAccountID != "account-123" {
		t.Errorf("ChatGPT-Account-ID = %q, want %q", gotAccountID, "account-123")
	}
	if gotBody["model"] != "gpt-5.5" {
		t.Errorf("model = %v, want gpt-5.5", gotBody["model"])
	}
	if gotBody["instructions"] != "be terse" {
		t.Errorf("instructions = %v, want be terse", gotBody["instructions"])
	}
	if gotBody["store"] != false {
		t.Errorf("store = %v, want false", gotBody["store"])
	}
	if gotBody["stream"] != true {
		t.Errorf("stream = %v, want true", gotBody["stream"])
	}
	if _, ok := gotBody["max_output_tokens"]; ok {
		t.Errorf("max_output_tokens should be omitted for ChatGPT auth, got %v", gotBody["max_output_tokens"])
	}
	if resp.Content() != "ok" {
		t.Errorf("Content() = %q, want %q", resp.Content(), "ok")
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 5 {
		t.Fatalf("Usage = %#v, want total 5", resp.Usage)
	}
}

func TestResponsesClient_MapsFunctionToolCalls(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"resp_tool",
			"model":"gpt-5.5",
			"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"README.md\"}"}]
		}`))
	}))
	defer server.Close()

	client := NewResponsesClient(ClientConfig{
		URL:    server.URL + "/v1",
		APIKey: "token",
		Model:  "gpt-5.5",
	})
	resp, err := client.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages: []Message{{Role: "user", Content: "read it"}},
		Tools: []ToolDef{{
			Type: "function",
			Function: FunctionDef{
				Name:        "read_file",
				Description: "read a file",
				Parameters:  map[string]any{"type": "object"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(calls))
	}
	if calls[0].ID != "call_1" {
		t.Errorf("tool call ID = %q, want call_1", calls[0].ID)
	}
	if calls[0].Function.Name != "read_file" {
		t.Errorf("tool name = %q, want read_file", calls[0].Function.Name)
	}
	if calls[0].Function.Arguments != `{"path":"README.md"}` {
		t.Errorf("arguments = %q", calls[0].Function.Arguments)
	}
}

func TestResponsesClient_MapsStreamCompletedEvent(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("event: response.output_text.delta\r\n"))
		_, _ = w.Write([]byte(`data: {"type":"response.output_text.delta","delta":"stream ok"}` + "\r\n\r\n"))
		_, _ = w.Write([]byte("event: response.completed\n"))
		_, _ = w.Write([]byte(`data: {"type":"response.completed","response":{"id":"resp_stream","model":"gpt-5.5","output":[],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}` + "\n\n"))
	}))
	defer server.Close()

	client := NewResponsesClient(ClientConfig{
		URL:    server.URL + "/v1",
		APIKey: "token",
		Model:  "gpt-5.5",
	})
	resp, err := client.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages: []Message{{Role: "user", Content: "ping"}},
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if resp.Content() != "stream ok" {
		t.Errorf("Content() = %q, want stream ok", resp.Content())
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 3 {
		t.Fatalf("Usage = %#v, want total 3", resp.Usage)
	}
}

func TestResponsesClient_DropsResponsesFunctionCallsWithoutName(t *testing.T) {
	raw := []byte(`{
		"id":"resp_tool",
		"model":"gpt-5.5",
		"output":[
			{"type":"function_call","id":"fc_named","call_id":"call_named","name":"code_search","arguments":"{\"search_text\":\"getvCheck\"}"},
			{"type":"function_call","id":"fc_empty","call_id":"fc_empty","name":"","arguments":"{\"search_text\":\"getvCheck\"}"}
		]
	}`)

	resp, err := mapResponsesResponse(raw)
	if err != nil {
		t.Fatalf("mapResponsesResponse: %v", err)
	}
	calls := resp.ToolCalls()
	if len(calls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1: %#v", len(calls), calls)
	}
	if calls[0].ID != "call_named" {
		t.Errorf("tool call ID = %q, want call_named", calls[0].ID)
	}
	if calls[0].Function.Name != "code_search" {
		t.Errorf("tool call name = %q, want code_search", calls[0].Function.Name)
	}
}

func TestResponsesClient_RejectsAssistantToolCallWithoutName(t *testing.T) {
	client := NewResponsesClient(ClientConfig{
		URL:    "https://api.example.com/v1",
		APIKey: "token",
		Model:  "gpt-5.5",
	})

	_, err := client.buildResponsesBody("gpt-5.5", ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "review"},
			NewToolCallMessage("", []ToolCall{{
				ID:   "fc_empty",
				Type: "function",
				Function: FunctionCall{
					Name:      "",
					Arguments: `{"search_text":"getvCheck"}`,
				},
			}}),
			NewToolResultMessage("fc_empty", "Error: Tool not found."),
		},
	})
	if err == nil {
		t.Fatal("buildResponsesBody succeeded, want missing tool name error")
	}
	if !strings.Contains(err.Error(), "missing function name") {
		t.Fatalf("error = %v, want missing function name", err)
	}
}

func TestBuildAnthropicParams_CacheControl(t *testing.T) {
	client := NewAnthropicClient(ClientConfig{URL: "https://api.anthropic.com"})

	req := ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a code reviewer."},
			{Role: "system", Content: "Be concise."},
			{Role: "user", Content: "Review this code."},
		},
		Tools: []ToolDef{
			{Type: "function", Function: FunctionDef{Name: "tool_a", Description: "first tool", Parameters: map[string]any{"type": "object"}}},
			{Type: "function", Function: FunctionDef{Name: "tool_b", Description: "second tool", Parameters: map[string]any{"type": "object"}}},
		},
	}

	params, err := client.buildAnthropicParams("claude-sonnet-4-20250514", req)
	if err != nil {
		t.Fatalf("buildAnthropicParams: %v", err)
	}

	t.Run("last system block has cache control", func(t *testing.T) {
		if len(params.System) < 2 {
			t.Fatalf("expected at least 2 system blocks, got %d", len(params.System))
		}
		last := params.System[len(params.System)-1]
		if last.CacheControl.Type != "ephemeral" {
			t.Errorf("last system block CacheControl.Type = %q, want %q", last.CacheControl.Type, "ephemeral")
		}
	})

	t.Run("non-last system block has no cache control", func(t *testing.T) {
		first := params.System[0]
		if first.CacheControl.Type != "" {
			t.Errorf("first system block CacheControl.Type = %q, want empty", first.CacheControl.Type)
		}
	})

	t.Run("last tool has cache control", func(t *testing.T) {
		if len(params.Tools) < 2 {
			t.Fatalf("expected at least 2 tools, got %d", len(params.Tools))
		}
		last := params.Tools[len(params.Tools)-1]
		if last.OfTool == nil {
			t.Fatal("last tool OfTool is nil")
		}
		if last.OfTool.CacheControl.Type != "ephemeral" {
			t.Errorf("last tool CacheControl.Type = %q, want %q", last.OfTool.CacheControl.Type, "ephemeral")
		}
	})

	t.Run("non-last tool has no cache control", func(t *testing.T) {
		first := params.Tools[0]
		if first.OfTool == nil {
			t.Fatal("first tool OfTool is nil")
		}
		if first.OfTool.CacheControl.Type != "" {
			t.Errorf("first tool CacheControl.Type = %q, want empty", first.OfTool.CacheControl.Type)
		}
	})

	t.Run("top-level CacheControl is not set", func(t *testing.T) {
		if params.CacheControl.Type != "" {
			t.Errorf("params.CacheControl.Type = %q, want empty", params.CacheControl.Type)
		}
	})
}

func TestBuildAnthropicParams_CacheControl_NoTools(t *testing.T) {
	client := NewAnthropicClient(ClientConfig{URL: "https://api.anthropic.com"})

	req := ChatRequest{
		Messages: []Message{
			{Role: "system", Content: "You are a planner."},
			{Role: "user", Content: "Plan the review."},
		},
	}

	params, err := client.buildAnthropicParams("claude-sonnet-4-20250514", req)
	if err != nil {
		t.Fatalf("buildAnthropicParams: %v", err)
	}

	if len(params.System) == 0 {
		t.Fatal("expected system blocks")
	}
	last := params.System[len(params.System)-1]
	if last.CacheControl.Type != "ephemeral" {
		t.Errorf("system CacheControl.Type = %q, want %q", last.CacheControl.Type, "ephemeral")
	}
	if len(params.Tools) != 0 {
		t.Errorf("expected no tools, got %d", len(params.Tools))
	}
}

func TestBuildAnthropicParams_CacheControl_NoSystem(t *testing.T) {
	client := NewAnthropicClient(ClientConfig{URL: "https://api.anthropic.com"})

	req := ChatRequest{
		Messages: []Message{
			{Role: "user", Content: "Hello"},
		},
		Tools: []ToolDef{
			{Type: "function", Function: FunctionDef{Name: "tool_a", Description: "a tool", Parameters: map[string]any{"type": "object"}}},
		},
	}

	params, err := client.buildAnthropicParams("claude-sonnet-4-20250514", req)
	if err != nil {
		t.Fatalf("buildAnthropicParams: %v", err)
	}

	if len(params.System) != 0 {
		t.Errorf("expected no system blocks, got %d", len(params.System))
	}
	if len(params.Tools) == 0 {
		t.Fatal("expected tools")
	}
	if params.Tools[0].OfTool.CacheControl.Type != "ephemeral" {
		t.Errorf("tool CacheControl.Type = %q, want %q", params.Tools[0].OfTool.CacheControl.Type, "ephemeral")
	}
}

func TestAnthropicClient_UsesConfiguredXAPIKeyHeader(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "env-oauth-token")

	var gotXAPIKey string
	var gotAuthorization string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("X-Api-Key")
		gotAuthorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"msg_test",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
	defer server.Close()

	client := NewAnthropicClient(ClientConfig{
		URL:        server.URL + "/v1/messages",
		APIKey:     "sk-ant-api03-test",
		Model:      "claude-test",
		AuthHeader: "x-api-key",
	})

	_, err := client.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "ping"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if gotXAPIKey != "sk-ant-api03-test" {
		t.Errorf("X-Api-Key = %q, want %q", gotXAPIKey, "sk-ant-api03-test")
	}
	if gotAuthorization != "" {
		t.Errorf("Authorization = %q, want empty", gotAuthorization)
	}
}

func TestAnthropicClient_UsesConfiguredAuthorizationHeader(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "env-api-key")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "")

	var gotXAPIKey string
	var gotAuthorization string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("X-Api-Key")
		gotAuthorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"msg_test",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
	defer server.Close()

	client := NewAnthropicClient(ClientConfig{
		URL:        server.URL + "/v1/messages",
		APIKey:     "oauth-token",
		Model:      "claude-test",
		AuthHeader: "authorization",
	})

	_, err := client.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "ping"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if gotAuthorization != "Bearer oauth-token" {
		t.Errorf("Authorization = %q, want %q", gotAuthorization, "Bearer oauth-token")
	}
	if gotXAPIKey != "" {
		t.Errorf("X-Api-Key = %q, want empty", gotXAPIKey)
	}
}

func TestAnthropicClient_DefaultsToAuthorizationHeader(t *testing.T) {
	var gotXAPIKey string
	var gotAuthorization string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotXAPIKey = r.Header.Get("X-Api-Key")
		gotAuthorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"id":"msg_test",
			"type":"message",
			"role":"assistant",
			"model":"claude-test",
			"content":[{"type":"text","text":"ok"}],
			"stop_reason":"end_turn",
			"usage":{"input_tokens":1,"output_tokens":1}
		}`))
	}))
	defer server.Close()

	client := NewAnthropicClient(ClientConfig{
		URL:    server.URL + "/v1/messages",
		APIKey: "oauth-token",
		Model:  "claude-test",
	})

	_, err := client.CompletionsWithCtx(context.Background(), ChatRequest{
		Messages:  []Message{{Role: "user", Content: "ping"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if gotAuthorization != "Bearer oauth-token" {
		t.Errorf("Authorization = %q, want %q", gotAuthorization, "Bearer oauth-token")
	}
	if gotXAPIKey != "" {
		t.Errorf("X-Api-Key = %q, want empty", gotXAPIKey)
	}
}

// Verify the SDK constant is accessible (compile-time check).
var _ anthropic.CacheControlEphemeralParam = anthropic.NewCacheControlEphemeralParam()
