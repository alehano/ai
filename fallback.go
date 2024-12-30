package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
)

type FallbackGen struct {
	gens          []LLMGen
	currentModel  string
	errorCallback func(error)
}

func NewFallbackGen(gens []LLMGen, errorCallback func(error)) *FallbackGen {
	return &FallbackGen{gens: gens, errorCallback: errorCallback}
}

func (f *FallbackGen) generateWithFallback(fn func(gen LLMGen) (string, error)) (string, error) {
	var lastErr error
	for _, gen := range f.gens {
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

func (f *FallbackGen) Generate(ctx context.Context, systemPrompt, prompt string) (string, error) {
	return f.generateWithFallback(func(gen LLMGen) (string, error) {
		return gen.Generate(ctx, systemPrompt, prompt)
	})
}

func (f *FallbackGen) GenerateStream(ctx context.Context, systemPrompt, prompt string, resultCh chan string, doneCh chan bool, errCh chan error) {
	var lastErr error
	for i, gen := range f.gens {
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

func (f *FallbackGen) GetModel() string {
	return f.currentModel
}

func (f *FallbackGen) GenerateFromImage(ctx context.Context, prompt string, image io.Reader, mimeType MimeType) (string, error) {
	return f.generateWithFallback(func(gen LLMGen) (string, error) {
		return gen.GenerateFromImage(ctx, prompt, image, mimeType)
	})
}

func (f *FallbackGen) GenerateFromImages(ctx context.Context, prompt string, images []io.Reader, mimeTypes []MimeType) (string, error) {
	return f.generateWithFallback(func(gen LLMGen) (string, error) {
		return gen.GenerateFromImages(ctx, prompt, images, mimeTypes)
	})
}

func (f *FallbackGen) GenerateFromChat(ctx context.Context, messages []Message) (string, error) {
	var lastErr error
	for _, gen := range f.gens {
		response, err := gen.GenerateFromChat(ctx, messages)
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
