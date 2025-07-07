package ai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"time"
)

// Helper function to copy timeout from original context and create a fresh context
func copyContextTimeout(originalCtx context.Context) (context.Context, context.CancelFunc) {
	if deadline, ok := originalCtx.Deadline(); ok {
		timeout := time.Until(deadline)
		if timeout <= 0 {
			// If deadline has passed, give a minimal timeout to avoid immediate cancellation
			timeout = time.Second
		}
		return context.WithTimeout(context.Background(), timeout)
	}
	// If no deadline, create a context that can be cancelled but has no timeout
	return context.WithCancel(context.Background())
}

type FallbackLLM struct {
	llms          []LLM
	currentModel  string
	errorCallback func(error)
}

func NewFallbackLLM(gens []LLM, errorCallback func(error)) *FallbackLLM {
	return &FallbackLLM{llms: gens, errorCallback: errorCallback}
}

func (f *FallbackLLM) generateWithFallback(originalCtx context.Context, fn func(ctx context.Context, gen LLM) (string, error)) (string, error) {
	var lastErr error
	for _, gen := range f.llms {
		// Create fresh context with same timeout for each LLM attempt
		ctx, cancel := copyContextTimeout(originalCtx)

		response, err := fn(ctx, gen)
		cancel() // Always cancel the context when done

		if err == nil {
			f.currentModel = gen.GetModel()
			return response, nil
		}
		if f.errorCallback != nil {
			f.errorCallback(fmt.Errorf("Model %s error: %v", gen.GetModel(), err))
		}
		lastErr = err
	}
	return "", fmt.Errorf("LLM failed, last error: %v", lastErr)
}

func (f *FallbackLLM) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	return f.generateWithFallback(ctx, func(freshCtx context.Context, gen LLM) (string, error) {
		return gen.Generate(freshCtx, systemPrompt, prompt)
	})
}

func (f *FallbackLLM) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	var lastErr error
	for i, gen := range f.llms {
		genLocal := gen // Create local copy for goroutine
		// Send [CLEAR] message if this is not the first generator
		if i > 0 {
			// Check if the original context was cancelled (not just timed out) before proceeding
			select {
			case <-ctx.Done():
				if ctx.Err() == context.Canceled {
					errCh <- ctx.Err()
					return
				}
				// If it's a timeout, continue with fallback
			default:
			}

			select {
			case resultCh <- "[CLEAR]":
			default:
				// Channel might be closed, continue anyway
			}
		}

		// Create fresh context with same timeout for this LLM attempt
		genCtx, genCancel := copyContextTimeout(ctx)
		genErrCh := make(chan error, 1)
		genDoneCh := make(chan bool, 1)

		go func() {
			// fmt.Printf("[Debug] Generating with model: %s\n", gen.GetModel())
			genLocal.GenerateStream(genCtx, systemPrompt, prompt, resultCh, genDoneCh, genErrCh)
		}()

		select {
		case <-genDoneCh:
			genCancel()
			f.currentModel = gen.GetModel() // Set the current model
			doneCh <- true
			return
		case err := <-genErrCh:
			genCancel()
			if err != nil {
				lastErr = err
				if f.errorCallback != nil {
					f.errorCallback(fmt.Errorf("Model %s error: %v", gen.GetModel(), err))
				}
				// Continue to the next generator for any error (including timeout)
			} else {
				// Wait for all results before returning
				<-genDoneCh
				doneCh <- true
				return
			}
		}

		// Check if user explicitly cancelled (not just timeout) between attempts
		select {
		case <-ctx.Done():
			if ctx.Err() == context.Canceled {
				genCancel()
				errCh <- ctx.Err()
				return
			}
			// If it's a timeout, continue to next LLM
		default:
		}
	}
	var finalErr error
	if lastErr != nil {
		finalErr = fmt.Errorf("LLM failed, last error: %v", lastErr)
	} else {
		finalErr = errors.New("LLM failed")
	}

	select {
	case errCh <- finalErr:
	case <-ctx.Done():
	}
}

func (f *FallbackLLM) GetModel() string {
	return f.currentModel
}

// Add a helper function to handle buffering of images
func bufferImage(image io.Reader) (*bytes.Buffer, error) {
	if image == nil {
		return nil, nil
	}
	buf := new(bytes.Buffer)
	if _, err := io.Copy(buf, image); err != nil {
		return nil, fmt.Errorf("failed to copy image data: %w", err)
	}
	return buf, nil
}

// Add a helper function to create new readers from buffers
func newReadersFromBuffers(bufs []*bytes.Buffer) []io.Reader {
	readers := make([]io.Reader, len(bufs))
	for i, buf := range bufs {
		if buf != nil {
			readers[i] = bytes.NewReader(buf.Bytes())
		}
	}
	return readers
}

func (f *FallbackLLM) GenerateWithImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	imageBuf, err := bufferImage(image)
	if err != nil {
		return "", err
	}

	return f.generateWithFallback(ctx, func(freshCtx context.Context, gen LLM) (string, error) {
		var currentImageReader io.Reader
		if imageBuf != nil {
			currentImageReader = bytes.NewReader(imageBuf.Bytes())
		}
		return gen.GenerateWithImage(freshCtx, prompt, currentImageReader, mimeType)
	})
}

func (f *FallbackLLM) GenerateWithImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	if len(images) != len(mimeTypes) {
		return "", fmt.Errorf("number of images (%d) does not match number of mime types (%d)", len(images), len(mimeTypes))
	}

	// Buffer all images at once
	imageBufs := make([]*bytes.Buffer, len(images))
	for i, img := range images {
		buf, err := bufferImage(img)
		if err != nil {
			return "", fmt.Errorf("failed to buffer image %d: %w", i, err)
		}
		imageBufs[i] = buf
	}

	return f.generateWithFallback(ctx, func(freshCtx context.Context, gen LLM) (string, error) {
		return gen.GenerateWithImages(freshCtx, prompt, newReadersFromBuffers(imageBufs), mimeTypes)
	})
}

func (f *FallbackLLM) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	var lastErr error
	for _, gen := range f.llms {
		// Create fresh context with same timeout for each LLM attempt
		freshCtx, cancel := copyContextTimeout(ctx)

		response, err := gen.GenerateWithMessages(freshCtx, messages)
		cancel() // Always cancel the context when done

		if err == nil {
			f.currentModel = gen.GetModel()
			return response, nil
		}
		if f.errorCallback != nil {
			f.errorCallback(fmt.Errorf("Model %s error: %v", gen.GetModel(), err))
		}
		lastErr = err
	}
	return "", fmt.Errorf("LLM failed, last error: %v", lastErr)
}
