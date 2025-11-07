use std::time::Duration;
use axum::{
    extract::WebSocketUpgrade,
    response::IntoResponse,
    routing::get,
    Router,
};
use axum::extract::ws::{Message, WebSocket};

use hft_optimisation::suboptimal::LOBSimulator;

#[tokio::main]
async fn main() {
    let app = Router::new().route("/ws", get(ws_handler));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
    println!("WebSocket simulator listening on ws://{}/ws", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    let mut sim = LOBSimulator::new();

    // 1) Bootstrap (full book)
    let boot = sim.bootstrap_update();
    let payload = serde_json::to_string(&boot).unwrap();
    if socket.send(Message::Text(payload)).await.is_err() {
        return;
    }

    // 2) Incremental stream
    loop {
        let upd = sim.next_update();

        // Tip: to avoid sending overly heavy packets, we can
        // filter and limit to N diffs; here we send everything for consistency.
        let json = serde_json::to_string(&upd).unwrap();

        if socket.send(Message::Text(json)).await.is_err() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(sim.dt_ms())).await;
    }
}
