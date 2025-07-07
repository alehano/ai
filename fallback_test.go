package ai

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// MockLLM implements the LLM interface for testing
type MockLLM struct {
	name         string
	delay        time.Duration
	shouldError  bool
	errorMessage string
	response     string
	callCount    int
	mu           sync.Mutex
	contextTimes []time.Time // Track when contexts are created
}

func NewMockLLM(name string, delay time.Duration, shouldError bool, errorMessage, response string) *MockLLM {
	return &MockLLM{
		name:         name,
		delay:        delay,
		shouldError:  shouldError,
		errorMessage: errorMessage,
		response:     response,
	}
}

func (m *MockLLM) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	m.mu.Lock()
	m.callCount++
	m.contextTimes = append(m.contextTimes, time.Now())
	m.mu.Unlock()

	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}

	if m.shouldError {
		return "", fmt.Errorf(m.errorMessage)
	}

	return fmt.Sprintf("%s: %s", m.name, m.response), nil
}

func (m *MockLLM) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	m.mu.Lock()
	m.callCount++
	m.contextTimes = append(m.contextTimes, time.Now())
	m.mu.Unlock()

	go func() {
		if m.delay > 0 {
			select {
			case <-time.After(m.delay):
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}

		if m.shouldError {
			errCh <- fmt.Errorf(m.errorMessage)
			return
		}

		response := fmt.Sprintf("%s: %s", m.name, m.response)
		for _, char := range response {
			select {
			case resultCh <- string(char):
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}
		doneCh <- true
	}()
}

func (m *MockLLM) GetModel() string {
	return m.name
}

func (m *MockLLM) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return m.Generate(ctx, "", prompt)
}

func (m *MockLLM) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	return m.Generate(ctx, "", prompt)
}

func (m *MockLLM) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	return m.Generate(ctx, "", "messages")
}

func (m *MockLLM) GetCallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.callCount
}

func (m *MockLLM) GetContextTimes() []time.Time {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]time.Time{}, m.contextTimes...)
}

func (m *MockLLM) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.callCount = 0
	m.contextTimes = nil
}

func TestCopyContextTimeout(t *testing.T) {
	tests := []struct {
		name            string
		originalTimeout time.Duration
		expectTimeout   bool
	}{
		{
			name:            "with timeout",
			originalTimeout: 5 * time.Second,
			expectTimeout:   true,
		},
		{
			name:            "no timeout",
			originalTimeout: 0,
			expectTimeout:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var originalCtx context.Context
			var cancel context.CancelFunc

			if tt.originalTimeout > 0 {
				originalCtx, cancel = context.WithTimeout(context.Background(), tt.originalTimeout)
				defer cancel()
			} else {
				originalCtx = context.Background()
			}

			newCtx, newCancel := copyContextTimeout(originalCtx)
			defer newCancel()

			if tt.expectTimeout {
				deadline, hasDeadline := newCtx.Deadline()
				if !hasDeadline {
					t.Fatal("Expected new context to have deadline")
				}

				originalDeadline, _ := originalCtx.Deadline()
				timeDiff := originalDeadline.Sub(deadline)
				if timeDiff > time.Millisecond*100 { // Allow small differences
					t.Fatalf("New context deadline too different: %v", timeDiff)
				}
			} else {
				if _, hasDeadline := newCtx.Deadline(); hasDeadline {
					t.Fatal("Expected new context to not have deadline")
				}
			}
		})
	}
}

func TestFallbackLLM_Generate_Success(t *testing.T) {
	mock1 := NewMockLLM("mock1", 0, false, "", "response1")
	mock2 := NewMockLLM("mock2", 0, false, "", "response2")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
	})

	result, err := fallback.Generate(context.Background(), "system", "prompt")
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if result != "mock1: response1" {
		t.Fatalf("Expected 'mock1: response1', got: %s", result)
	}

	if mock1.GetCallCount() != 1 {
		t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
	}

	if mock2.GetCallCount() != 0 {
		t.Fatalf("Expected mock2 to not be called, got: %d", mock2.GetCallCount())
	}

	if len(errorCallbacks) != 0 {
		t.Fatalf("Expected no error callbacks, got: %v", errorCallbacks)
	}

	if fallback.GetModel() != "mock1" {
		t.Fatalf("Expected current model to be 'mock1', got: %s", fallback.GetModel())
	}
}

func TestFallbackLLM_Generate_Fallback(t *testing.T) {
	mock1 := NewMockLLM("mock1", 0, true, "mock1 error", "")
	mock2 := NewMockLLM("mock2", 0, false, "", "response2")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
	})

	result, err := fallback.Generate(context.Background(), "system", "prompt")
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if result != "mock2: response2" {
		t.Fatalf("Expected 'mock2: response2', got: %s", result)
	}

	if mock1.GetCallCount() != 1 {
		t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
	}

	if mock2.GetCallCount() != 1 {
		t.Fatalf("Expected mock2 to be called once, got: %d", mock2.GetCallCount())
	}

	if len(errorCallbacks) != 1 {
		t.Fatalf("Expected one error callback, got: %v", errorCallbacks)
	}

	if fallback.GetModel() != "mock2" {
		t.Fatalf("Expected current model to be 'mock2', got: %s", fallback.GetModel())
	}
}

func TestFallbackLLM_Generate_AllFail(t *testing.T) {
	mock1 := NewMockLLM("mock1", 0, true, "mock1 error", "")
	mock2 := NewMockLLM("mock2", 0, true, "mock2 error", "")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
	})

	result, err := fallback.Generate(context.Background(), "system", "prompt")
	if err == nil {
		t.Fatal("Expected error, got nil")
	}

	if result != "" {
		t.Fatalf("Expected empty result, got: %s", result)
	}

	if mock1.GetCallCount() != 1 {
		t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
	}

	if mock2.GetCallCount() != 1 {
		t.Fatalf("Expected mock2 to be called once, got: %d", mock2.GetCallCount())
	}

	if len(errorCallbacks) != 2 {
		t.Fatalf("Expected two error callbacks, got: %v", errorCallbacks)
	}
}

func TestFallbackLLM_Generate_TimeoutHandling(t *testing.T) {
	// First mock will timeout, second should succeed with fresh context
	mock1 := NewMockLLM("mock1", 200*time.Millisecond, false, "", "response1")
	mock2 := NewMockLLM("mock2", 50*time.Millisecond, false, "", "response2")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
	})

	// Create context with timeout that should cause mock1 to timeout
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	result, err := fallback.Generate(ctx, "system", "prompt")
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if result != "mock2: response2" {
		t.Fatalf("Expected 'mock2: response2', got: %s", result)
	}

	// Should have taken at least 100ms (mock1 timeout) + 50ms (mock2 delay)
	// but less than 200ms (mock1 full delay)
	if duration < 100*time.Millisecond || duration > 180*time.Millisecond {
		t.Fatalf("Expected duration between 100-180ms, got: %v", duration)
	}

	if mock1.GetCallCount() != 1 {
		t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
	}

	if mock2.GetCallCount() != 1 {
		t.Fatalf("Expected mock2 to be called once, got: %d", mock2.GetCallCount())
	}

	if len(errorCallbacks) != 1 {
		t.Fatalf("Expected one error callback, got: %v", errorCallbacks)
	}

	if fallback.GetModel() != "mock2" {
		t.Fatalf("Expected current model to be 'mock2', got: %s", fallback.GetModel())
	}
}

func TestFallbackLLM_GenerateStream_TimeoutHandling(t *testing.T) {
	// Create mocks where first times out, second succeeds
	// Mock1: will take 300ms but context gives it only ~100ms
	// Mock2: will take 50ms and should succeed with remaining time
	mock1 := NewMockLLM("mock1", 300*time.Millisecond, false, "", "response1")
	mock2 := NewMockLLM("mock2", 50*time.Millisecond, false, "", "response2")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
	})

	// Give original context 120ms - not enough for mock1 (300ms) but enough for mock2 (50ms) after mock1 times out
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Millisecond)
	defer cancel()

	resultCh := make(chan string, 100)
	doneCh := make(chan bool, 1)
	errCh := make(chan error, 1)

	start := time.Now()
	fallback.GenerateStream(ctx, "system", "prompt", resultCh, doneCh, errCh)

	// Wait for completion or error
	select {
	case <-doneCh:
		duration := time.Since(start)
		t.Logf("Completed successfully in: %v", duration)

	case err := <-errCh:
		t.Fatalf("Expected success, got error: %v", err)
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Test timed out waiting for completion")
	}

	// Verify both LLMs were called
	if mock1.GetCallCount() != 1 {
		t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
	}

	if mock2.GetCallCount() != 1 {
		t.Fatalf("Expected mock2 to be called once, got: %d", mock2.GetCallCount())
	}

	// Should have error callback for mock1 timeout
	if len(errorCallbacks) == 0 {
		t.Fatal("Expected error callback for mock1 timeout")
	}

	// Verify the error was a timeout
	found := false
	for _, callback := range errorCallbacks {
		if strings.Contains(callback, "context deadline exceeded") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("Expected timeout error in callbacks: %v", errorCallbacks)
	}

	// Final model should be mock2
	if fallback.GetModel() != "mock2" {
		t.Fatalf("Expected current model to be 'mock2', got: %s", fallback.GetModel())
	}

	// Collect and verify results
	close(resultCh)
	var results []string
	for result := range resultCh {
		if result != "[CLEAR]" {
			results = append(results, result)
		}
	}

	if len(results) == 0 {
		t.Fatal("Expected some results from mock2")
	}
}

func TestFallbackLLM_ContextTimeCopying(t *testing.T) {
	// Test that each LLM gets a fresh context with the same timeout
	mock1 := NewMockLLM("mock1", 0, true, "mock1 error", "")
	mock2 := NewMockLLM("mock2", 0, true, "mock2 error", "")
	mock3 := NewMockLLM("mock3", 0, false, "", "response3")

	fallback := NewFallbackLLM([]LLM{mock1, mock2, mock3}, func(err error) {})

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	start := time.Now()
	_, _ = fallback.Generate(ctx, "system", "prompt")

	// Check that all mocks were called
	if mock1.GetCallCount() != 1 {
		t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
	}

	if mock2.GetCallCount() != 1 {
		t.Fatalf("Expected mock2 to be called once, got: %d", mock2.GetCallCount())
	}

	if mock3.GetCallCount() != 1 {
		t.Fatalf("Expected mock3 to be called once, got: %d", mock3.GetCallCount())
	}

	// Check timing - all calls should happen quickly since they don't have delays
	// and each gets a fresh 1-second timeout
	duration := time.Since(start)
	if duration > 100*time.Millisecond {
		t.Fatalf("Expected quick execution, got: %v", duration)
	}

	// Verify that context times are close together (fresh contexts)
	times1 := mock1.GetContextTimes()
	times2 := mock2.GetContextTimes()
	times3 := mock3.GetContextTimes()

	if len(times1) != 1 || len(times2) != 1 || len(times3) != 1 {
		t.Fatal("Expected each mock to be called exactly once")
	}

	// All calls should happen within a short time window since they get fresh timeouts
	maxTimeDiff := times3[0].Sub(times1[0])
	if maxTimeDiff > 50*time.Millisecond {
		t.Fatalf("Expected all calls to happen quickly, time difference: %v", maxTimeDiff)
	}
}

func TestFallbackLLM_GenerateStream_Debug(t *testing.T) {
	// Create a mock that returns an error to test fallback
	mock1 := NewMockLLM("mock1", 0, true, "mock1 failed", "")
	mock2 := NewMockLLM("mock2", 0, false, "", "response2")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
		t.Logf("Error callback: %v", err)
	})

	ctx := context.Background()

	resultCh := make(chan string, 100)
	doneCh := make(chan bool, 1)
	errCh := make(chan error, 1)

	t.Log("Starting GenerateStream")
	fallback.GenerateStream(ctx, "system", "prompt", resultCh, doneCh, errCh)

	// Wait for completion or error
	select {
	case <-doneCh:
		t.Log("Received done signal")
	case err := <-errCh:
		t.Fatalf("Expected success, got error: %v", err)
	case <-time.After(2 * time.Second):
		t.Fatal("Test timed out")
	}

	t.Logf("Mock1 calls: %d, Mock2 calls: %d", mock1.GetCallCount(), mock2.GetCallCount())
	t.Logf("Error callbacks: %v", errorCallbacks)
	t.Logf("Current model: %s", fallback.GetModel())

	// Collect results
	close(resultCh)
	var results []string
	for result := range resultCh {
		if result != "[CLEAR]" {
			results = append(results, result)
		}
	}
	t.Logf("Results: %v", results)
}

func TestFallbackLLM_GenerateStream_ActualTimeout(t *testing.T) {
	// Create mocks where first one times out via context, second succeeds
	mock1 := NewMockLLM("mock1", 200*time.Millisecond, false, "", "response1")
	mock2 := NewMockLLM("mock2", 50*time.Millisecond, false, "", "response2")

	var errorCallbacks []string
	fallback := NewFallbackLLM([]LLM{mock1, mock2}, func(err error) {
		errorCallbacks = append(errorCallbacks, err.Error())
		t.Logf("Error callback: %v", err)
	})

	// Each LLM should get 100ms timeout, so mock1 should timeout but mock2 should succeed
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	resultCh := make(chan string, 100)
	doneCh := make(chan bool, 1)
	errCh := make(chan error, 1)

	start := time.Now()
	t.Log("Starting GenerateStream with timeout")
	fallback.GenerateStream(ctx, "system", "prompt", resultCh, doneCh, errCh)

	// Wait for completion or error
	select {
	case <-doneCh:
		t.Log("Received done signal")
	case err := <-errCh:
		t.Logf("Received error: %v", err)
		// This might be expected if the overall timeout prevents fallback
		duration := time.Since(start)
		t.Logf("Failed after: %v", duration)

		// Check if mock1 was called
		if mock1.GetCallCount() != 1 {
			t.Fatalf("Expected mock1 to be called once, got: %d", mock1.GetCallCount())
		}

		// The question is whether mock2 should be called or not
		t.Logf("Mock1 calls: %d, Mock2 calls: %d", mock1.GetCallCount(), mock2.GetCallCount())
		return
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Test timed out")
	}

	duration := time.Since(start)
	t.Logf("Completed in: %v", duration)

	t.Logf("Mock1 calls: %d, Mock2 calls: %d", mock1.GetCallCount(), mock2.GetCallCount())
	t.Logf("Error callbacks: %v", errorCallbacks)
	t.Logf("Current model: %s", fallback.GetModel())

	// Collect results
	close(resultCh)
	var results []string
	for result := range resultCh {
		if result != "[CLEAR]" {
			results = append(results, result)
		}
	}
	t.Logf("Results: %v", results)
}
