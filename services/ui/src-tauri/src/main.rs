#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Manager,
};

#[tauri::command]
async fn get_system_status() -> Result<serde_json::Value, String> {
    use std::time::Duration;

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())?;

    let services = vec![
        ("broker", "http://localhost:8001/health"),
        ("llm_engine", "http://localhost:8002/health"),
        ("memory", "http://localhost:8003/health"),
        ("agent_core", "http://localhost:8000/health"),
        ("tool_system", "http://localhost:8004/health"),
        ("voice", "http://localhost:8005/health"),
    ];

    let mut status = serde_json::Map::new();
    for (name, url) in services {
        let ok = client.get(url).send().await.map(|r| r.status().is_success()).unwrap_or(false);
        status.insert(name.to_string(), serde_json::Value::Bool(ok));
    }

    Ok(serde_json::Value::Object(status))
}

#[tauri::command]
fn show_notification(title: &str, body: &str) {
    println!("Notification: [{title}] {body}");
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .invoke_handler(tauri::generate_handler![get_system_status, show_notification])
        .setup(|app| {
            let quit_item = MenuItem::with_id(app, "quit", "Quit JARVIS", true, None::<&str>)?;
            let show_item = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_item, &quit_item])?;

            let _tray = TrayIconBuilder::new()
                .menu(&menu)
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "quit" => app.exit(0),
                    "show" => {
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running JARVIS desktop app");
}
