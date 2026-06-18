package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"

	"github.com/open-code-review/open-code-review/internal/agent"
	"github.com/open-code-review/open-code-review/internal/model"
)

func TestSanitizeTerminal(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"plain text unchanged", "hello world", "hello world"},
		{"preserves tab", "col1\tcol2", "col1\tcol2"},
		{"preserves newline", "line1\nline2", "line1\nline2"},
		{"strips ESC", "before\x1b[2Jafter", "before[2Jafter"},
		{"strips OSC 52", "\x1b]52;c;dGVzdA==\x07", "]52;c;dGVzdA=="},
		{"strips BEL alone", "beep\x07done", "beepdone"},
		{"strips null byte", "a\x00b", "ab"},
		{"strips DEL", "a\x7fb", "ab"},
		{"strips carriage return", "fake\rreal", "fakereal"},
		{"empty string", "", ""},
		{"only control chars", "\x1b\x07\x00\x7f", ""},
		{"unicode preserved", "代码审查 レビュー 🔍", "代码审查 レビュー 🔍"},
		{"mixed safe and unsafe", "path\x1b[0m/file.go", "path[0m/file.go"},
		{"strips C1 CSI (U+009B)", "beforeafter", "beforeafter"},
		{"strips C1 OSC (U+009D)", "beforeafter", "beforeafter"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sanitizeTerminal(tt.in)
			if got != tt.want {
				t.Errorf("sanitizeTerminal(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestOutputTextWithWarningsPrintsSubtaskErrors(t *testing.T) {
	stdoutText, stderrText := captureOutput(t, func() {
		outputTextWithWarnings([]model.LlmComment{}, []agent.AgentWarning{{
			Type:    "subtask_error",
			File:    "confd/templates/collector.sh.tmpl",
			Message: "LLM completion error: responses API returned 400 Bad Request",
		}})
	})

	if !strings.Contains(stdoutText, "Some files could not be reviewed due to errors") {
		t.Fatalf("stdout = %q, want subtask error summary", stdoutText)
	}
	if !strings.Contains(stderrText, "[ocr] WARNING [subtask_error] confd/templates/collector.sh.tmpl") {
		t.Fatalf("stderr = %q, want subtask warning", stderrText)
	}
	if !strings.Contains(stderrText, "responses API returned 400 Bad Request") {
		t.Fatalf("stderr = %q, want underlying error", stderrText)
	}
}

func captureOutput(t *testing.T, fn func()) (string, string) {
	t.Helper()
	oldStdout := os.Stdout
	oldStderr := os.Stderr
	stdoutR, stdoutW, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe stdout: %v", err)
	}
	stderrR, stderrW, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe stderr: %v", err)
	}

	os.Stdout = stdoutW
	os.Stderr = stderrW
	defer func() {
		os.Stdout = oldStdout
		os.Stderr = oldStderr
	}()

	fn()
	_ = stdoutW.Close()
	_ = stderrW.Close()

	var stdoutBuf, stderrBuf bytes.Buffer
	_, _ = io.Copy(&stdoutBuf, stdoutR)
	_, _ = io.Copy(&stderrBuf, stderrR)
	_ = stdoutR.Close()
	_ = stderrR.Close()

	return stdoutBuf.String(), stderrBuf.String()
}
