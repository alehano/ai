package ai

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
)

type FallbackLLM struct {
	llms          []LLM
	currentModel  string
	errorCallback func(error)
}

func NewFallbackLLM(gens []LLM, errorCallback func(error)) *FallbackLLM {
	return &FallbackLLM{llms: gens, errorCallback: errorCallback}
}

func (f *FallbackLLM) generateWithFallback(fn func(gen LLM) (string, error)) (string, error) {
	var lastErr error
	for _, gen := range f.llms {
		response, err := fn(gen)
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
	return f.generateWithFallback(func(gen LLM) (string, error) {
		return gen.Generate(ctx, systemPrompt, prompt)
	})
}

func (f *FallbackLLM) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	var lastErr error
	for i, gen := range f.llms {
		// Send [CLEAR] message if this is not the first generator
		if i > 0 {
			select {
			case resultCh <- "[CLEAR]":
			case <-ctx.Done():
				errCh <- ctx.Err()
				return
			}
		}

		select {
		case <-ctx.Done():
			errCh <- ctx.Err()
			return
		default:
			genCtx, cancel := context.WithCancel(ctx)
			genErrCh := make(chan error, 1)
			genDoneCh := make(chan bool, 1)

			go func() {
				// fmt.Printf("[Debug] Generating with model: %s\n", gen.GetModel())
				gen.GenerateStream(genCtx, systemPrompt, prompt, resultCh, genDoneCh, genErrCh)
			}()

			select {
			case <-genDoneCh:
				cancel()
				f.currentModel = gen.GetModel() // Set the current model
				doneCh <- true
				return
			case err := <-genErrCh:
				cancel()
				if err == context.Canceled {
					errCh <- err
					return
				}
				if err != nil {
					lastErr = err
					f.errorCallback(fmt.Errorf("Model %s error: %v", gen.GetModel(), err))
					// Continue to the next generator
				} else {
					// Wait for all results before returning
					<-genDoneCh
					doneCh <- true
					return
				}
			case <-ctx.Done():
				cancel()
				errCh <- ctx.Err()
				return
			}
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

	return f.generateWithFallback(func(gen LLM) (string, error) {
		var currentImageReader io.Reader
		if imageBuf != nil {
			currentImageReader = bytes.NewReader(imageBuf.Bytes())
		}
		return gen.GenerateWithImage(ctx, prompt, currentImageReader, mimeType)
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

	return f.generateWithFallback(func(gen LLM) (string, error) {
		return gen.GenerateWithImages(ctx, prompt, newReadersFromBuffers(imageBufs), mimeTypes)
	})
}

func (f *FallbackLLM) GenerateWithMessages(ctx context.Context, messages []Message) (string, error) {
	var lastErr error
	for _, gen := range f.llms {
		response, err := gen.GenerateWithMessages(ctx, messages)
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
