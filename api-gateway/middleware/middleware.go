// Package middleware contains custom Fiber middleware for the gateway.
package middleware

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"

	"github.com/gofiber/fiber/v2"
)

// RequestID attaches a unique X-Request-ID header to every request/response.
func RequestID() fiber.Handler {
	return func(c *fiber.Ctx) error {
		id := c.Get("X-Request-ID")
		if id == "" {
			id = newID()
		}
		c.Set("X-Request-ID", id)
		c.Locals("request_id", id)
		return c.Next()
	}
}

func newID() string {
	b := make([]byte, 8)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// RateLimit is a small in-memory token-bucket per remote IP.
type bucket struct {
	tokens   int
	lastFill time.Time
}

type limiter struct {
	mu      sync.Mutex
	buckets map[string]*bucket
	max     int
	window  time.Duration
}

func RateLimit(maxPerWindow int, window time.Duration) fiber.Handler {
	l := &limiter{
		buckets: make(map[string]*bucket),
		max:     maxPerWindow,
		window:  window,
	}
	return func(c *fiber.Ctx) error {
		ip := c.IP()
		l.mu.Lock()
		b, ok := l.buckets[ip]
		now := time.Now()
		if !ok {
			b = &bucket{tokens: l.max, lastFill: now}
			l.buckets[ip] = b
		}
		// Refill if window expired.
		if now.Sub(b.lastFill) >= l.window {
			b.tokens = l.max
			b.lastFill = now
		}
		if b.tokens <= 0 {
			l.mu.Unlock()
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{
				"error": "rate limit exceeded",
			})
		}
		b.tokens--
		l.mu.Unlock()
		return c.Next()
	}
}
