// High-performance API gateway routing requests to the Python AI core.
//
// Run from the api-gateway directory:
//
//	go mod tidy
//	go run .
package main

import (
	"log"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/joho/godotenv"

	"github.com/orgio111/ai_agent/api-gateway/middleware"
	"github.com/orgio111/ai_agent/api-gateway/routes"
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func main() {
	// Best-effort load of repo-root .env
	_ = godotenv.Load("../.env")
	_ = godotenv.Load(".env")

	app := fiber.New(fiber.Config{
		AppName:               "Hybrid AI Gateway",
		ReadTimeout:           30 * time.Second,
		WriteTimeout:          120 * time.Second,
		IdleTimeout:           60 * time.Second,
		DisableStartupMessage: false,
		Prefork:               false,
	})

	app.Use(recover.New())
	app.Use(logger.New(logger.Config{
		Format: "[${time}] ${status} ${method} ${path} (${latency})\n",
	}))
	app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET,POST,PUT,DELETE,OPTIONS",
		AllowHeaders: "Content-Type,Authorization,X-Request-ID",
	}))
	app.Use(middleware.RequestID())
	app.Use(middleware.RateLimit(60, time.Minute))

	aiCoreURL := envOr("AI_CORE_URL", "http://localhost:8000")
	routes.Register(app, aiCoreURL)

	host := envOr("GATEWAY_HOST", "0.0.0.0")
	port := envOr("GATEWAY_PORT", "9000")
	addr := host + ":" + port

	log.Printf("Hybrid AI Gateway listening on %s → core %s", addr, aiCoreURL)
	if err := app.Listen(addr); err != nil {
		log.Fatalf("gateway failed: %v", err)
	}
}
