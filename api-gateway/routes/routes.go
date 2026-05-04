// Package routes wires HTTP routes to the AI core proxy.
package routes

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gofiber/fiber/v2"
)

const proxyTimeout = 180 * time.Second

// Register attaches all gateway routes to the Fiber app.
func Register(app *fiber.App, aiCoreURL string) {
	app.Get("/", indexHandler(aiCoreURL))
	app.Get("/health", healthHandler(aiCoreURL))

	api := app.Group("/api/v1")
	api.Post("/chat", proxyJSON(aiCoreURL, "/chat"))
	api.Post("/chat/stream", proxyStream(aiCoreURL, "/chat/stream"))
	api.Post("/agent/run", proxyJSON(aiCoreURL, "/agent/run"))
	api.Post("/memory/store", proxyJSON(aiCoreURL, "/memory/store"))
	api.Post("/memory/search", proxyJSON(aiCoreURL, "/memory/search"))
	api.Get("/tools", proxyGet(aiCoreURL, "/tools"))
	api.Post("/tools/run", proxyJSON(aiCoreURL, "/tools/run"))
}

func indexHandler(coreURL string) fiber.Handler {
	return func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"service": "hybrid-ai-gateway",
			"version": "1.0.0",
			"core":    coreURL,
			"endpoints": []string{
				"/health",
				"/api/v1/chat",
				"/api/v1/chat/stream",
				"/api/v1/agent/run",
				"/api/v1/memory/store",
				"/api/v1/memory/search",
				"/api/v1/tools",
				"/api/v1/tools/run",
			},
		})
	}
}

func healthHandler(coreURL string) fiber.Handler {
	client := &http.Client{Timeout: 5 * time.Second}
	return func(c *fiber.Ctx) error {
		resp, err := client.Get(coreURL + "/health")
		if err != nil {
			return c.Status(fiber.StatusServiceUnavailable).JSON(fiber.Map{
				"gateway": "ok",
				"core":    "unreachable",
				"error":   err.Error(),
			})
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		var coreStatus map[string]any
		_ = json.Unmarshal(body, &coreStatus)
		return c.JSON(fiber.Map{
			"gateway": "ok",
			"core":    coreStatus,
		})
	}
}

// proxyJSON forwards a JSON POST body to the AI core and returns its JSON.
func proxyJSON(coreURL, path string) fiber.Handler {
	client := &http.Client{Timeout: proxyTimeout}
	return func(c *fiber.Ctx) error {
		ctx, cancel := context.WithTimeout(c.Context(), proxyTimeout)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, http.MethodPost,
			coreURL+path, bytes.NewReader(c.Body()))
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		req.Header.Set("Content-Type", "application/json")
		if rid, ok := c.Locals("request_id").(string); ok {
			req.Header.Set("X-Request-ID", rid)
		}

		resp, err := client.Do(req)
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		defer resp.Body.Close()
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		c.Status(resp.StatusCode)
		c.Set("Content-Type", resp.Header.Get("Content-Type"))
		return c.Send(body)
	}
}

func proxyGet(coreURL, path string) fiber.Handler {
	client := &http.Client{Timeout: 30 * time.Second}
	return func(c *fiber.Ctx) error {
		req, err := http.NewRequestWithContext(c.Context(), http.MethodGet, coreURL+path, nil)
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		resp, err := client.Do(req)
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		defer resp.Body.Close()
		body, _ := io.ReadAll(resp.Body)
		c.Status(resp.StatusCode)
		c.Set("Content-Type", resp.Header.Get("Content-Type"))
		return c.Send(body)
	}
}

// proxyStream forwards a request and streams the response chunked.
func proxyStream(coreURL, path string) fiber.Handler {
	return func(c *fiber.Ctx) error {
		req, err := http.NewRequestWithContext(c.Context(), http.MethodPost,
			coreURL+path, bytes.NewReader(c.Body()))
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{Timeout: proxyTimeout}
		resp, err := client.Do(req)
		if err != nil {
			return jsonError(c, fiber.StatusBadGateway, err)
		}
		c.Status(resp.StatusCode)
		c.Set("Content-Type", "text/plain; charset=utf-8")
		c.Set("Cache-Control", "no-cache")
		c.Set("X-Accel-Buffering", "no")

		// Stream chunks to the client.
		c.Context().SetBodyStreamWriter(func(w *bufio.Writer) {
			defer resp.Body.Close()
			buf := make([]byte, 1024)
			for {
				n, rerr := resp.Body.Read(buf)
				if n > 0 {
					if _, werr := w.Write(buf[:n]); werr != nil {
						return
					}
					if ferr := w.Flush(); ferr != nil {
						return
					}
				}
				if rerr != nil {
					return
				}
			}
		})
		return nil
	}
}

func jsonError(c *fiber.Ctx, status int, err error) error {
	return c.Status(status).JSON(fiber.Map{
		"error":   "gateway_proxy_error",
		"message": fmt.Sprintf("%v", err),
	})
}
