//! Logging wrapper for LLM drivers.
//!
//! Wraps any `LlmDriver` implementation to emit structured tracing logs
//! at `info` (summary), `debug` (details), and `trace` (full bodies) levels
//! for every LLM request/response cycle.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace};

/// A wrapper around any `LlmDriver` that logs request/response details.
pub struct LoggingDriver {
    inner: Arc<dyn LlmDriver>,
    provider: String,
    /// Maximum characters for trace-level body logging.
    body_max_chars: usize,
}

impl LoggingDriver {
    /// Create a new logging wrapper.
    ///
    /// - `inner`: the underlying LLM driver
    /// - `provider`: provider name for log fields (e.g. "openai", "anthropic")
    /// - `body_max_chars`: max chars to log for request/response bodies at trace level
    pub fn new(inner: Arc<dyn LlmDriver>, provider: String, body_max_chars: usize) -> Self {
        Self {
            inner,
            provider,
            body_max_chars,
        }
    }

    /// Truncate a string to `body_max_chars`, appending `...[truncated]` if needed.
    fn truncate_body(&self, s: &str) -> String {
        if s.len() <= self.body_max_chars {
            s.to_string()
        } else {
            // Find a char boundary near the limit to avoid splitting multi-byte chars
            let end = s
                .char_indices()
                .take_while(|(i, _)| *i < self.body_max_chars)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            format!("{}...[truncated, total {} chars]", &s[..end], s.len())
        }
    }

    /// Format messages summary for debug logging.
    fn messages_summary(messages: &[openfang_types::message::Message]) -> String {
        use openfang_types::message::Role;
        let mut user = 0u32;
        let mut assistant = 0u32;
        let mut system = 0u32;
        for m in messages {
            match m.role {
                Role::User => user += 1,
                Role::Assistant => assistant += 1,
                Role::System => system += 1,
            }
        }
        format!(
            "total={} user={} assistant={} system={}",
            messages.len(),
            user,
            assistant,
            system,
        )
    }

    /// Serialize messages to a string for trace-level logging.
    fn messages_body(messages: &[openfang_types::message::Message]) -> String {
        serde_json::to_string(messages).unwrap_or_else(|_| "<serialization failed>".to_string())
    }
}

#[async_trait]
impl LlmDriver for LoggingDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let model = request.model.clone();
        let msg_count = request.messages.len();
        let tool_count = request.tools.len();
        let system_len = request.system.as_ref().map_or(0, |s| s.len());

        info!(
            provider = %self.provider,
            model = %model,
            messages = msg_count,
            tools = tool_count,
            max_tokens = request.max_tokens,
            temperature = request.temperature,
            system_prompt_len = system_len,
            "LLM request"
        );

        debug!(
            provider = %self.provider,
            model = %model,
            messages_detail = %Self::messages_summary(&request.messages),
            tool_names = %request.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(","),
            "LLM request details"
        );

        trace!(
            provider = %self.provider,
            model = %model,
            messages_body = %self.truncate_body(&Self::messages_body(&request.messages)),
            system_prompt = %self.truncate_body(request.system.as_deref().unwrap_or("")),
            "LLM request body"
        );

        let start = Instant::now();
        let result = self.inner.complete(request).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        match &result {
            Ok(response) => {
                let text_len = response.text().len();
                info!(
                    provider = %self.provider,
                    model = %model,
                    elapsed_ms = elapsed_ms,
                    stop_reason = ?response.stop_reason,
                    input_tokens = response.usage.input_tokens,
                    output_tokens = response.usage.output_tokens,
                    response_text_len = text_len,
                    tool_calls = response.tool_calls.len(),
                    "LLM response"
                );

                if !response.tool_calls.is_empty() {
                    debug!(
                        provider = %self.provider,
                        model = %model,
                        tool_call_names = %response.tool_calls.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(","),
                        "LLM response tool calls"
                    );
                }

                trace!(
                    provider = %self.provider,
                    model = %model,
                    response_body = %self.truncate_body(&response.text()),
                    "LLM response body"
                );
            }
            Err(e) => {
                error!(
                    provider = %self.provider,
                    model = %model,
                    elapsed_ms = elapsed_ms,
                    error = %e,
                    "LLM request failed"
                );
            }
        }

        result
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let model = request.model.clone();
        let msg_count = request.messages.len();
        let tool_count = request.tools.len();
        let system_len = request.system.as_ref().map_or(0, |s| s.len());

        info!(
            provider = %self.provider,
            model = %model,
            messages = msg_count,
            tools = tool_count,
            max_tokens = request.max_tokens,
            temperature = request.temperature,
            system_prompt_len = system_len,
            streaming = true,
            "LLM request"
        );

        debug!(
            provider = %self.provider,
            model = %model,
            messages_detail = %Self::messages_summary(&request.messages),
            tool_names = %request.tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(","),
            "LLM stream request details"
        );

        trace!(
            provider = %self.provider,
            model = %model,
            messages_body = %self.truncate_body(&Self::messages_body(&request.messages)),
            system_prompt = %self.truncate_body(request.system.as_deref().unwrap_or("")),
            "LLM stream request body"
        );

        let start = Instant::now();
        let result = self.inner.stream(request, tx).await;
        let elapsed_ms = start.elapsed().as_millis() as u64;

        match &result {
            Ok(response) => {
                let text_len = response.text().len();
                info!(
                    provider = %self.provider,
                    model = %model,
                    elapsed_ms = elapsed_ms,
                    stop_reason = ?response.stop_reason,
                    input_tokens = response.usage.input_tokens,
                    output_tokens = response.usage.output_tokens,
                    response_text_len = text_len,
                    tool_calls = response.tool_calls.len(),
                    streaming = true,
                    "LLM response"
                );

                trace!(
                    provider = %self.provider,
                    model = %model,
                    response_body = %self.truncate_body(&response.text()),
                    "LLM stream response body"
                );
            }
            Err(e) => {
                error!(
                    provider = %self.provider,
                    model = %model,
                    elapsed_ms = elapsed_ms,
                    error = %e,
                    streaming = true,
                    "LLM stream request failed"
                );
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_driver::CompletionResponse;
    use openfang_types::message::{ContentBlock, StopReason, TokenUsage};

    struct FakeDriver;

    #[async_trait]
    impl LlmDriver for FakeDriver {
        async fn complete(
            &self,
            _request: CompletionRequest,
        ) -> Result<CompletionResponse, LlmError> {
            Ok(CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: "Hello from fake driver!".to_string(),
                }],
                stop_reason: StopReason::EndTurn,
                tool_calls: vec![],
                usage: TokenUsage {
                    input_tokens: 10,
                    output_tokens: 5,
                },
            })
        }
    }

    #[test]
    fn test_truncate_body_short() {
        let driver = LoggingDriver::new(
            Arc::new(FakeDriver),
            "test".to_string(),
            100,
        );
        let short = "hello world";
        assert_eq!(driver.truncate_body(short), "hello world");
    }

    #[test]
    fn test_truncate_body_long() {
        let driver = LoggingDriver::new(
            Arc::new(FakeDriver),
            "test".to_string(),
            10,
        );
        let long = "abcdefghijklmnopqrstuvwxyz";
        let truncated = driver.truncate_body(long);
        assert!(truncated.starts_with("abcdefghij"));
        assert!(truncated.contains("[truncated"));
        assert!(truncated.contains("26 chars"));
    }

    #[test]
    fn test_truncate_body_multibyte() {
        let driver = LoggingDriver::new(
            Arc::new(FakeDriver),
            "test".to_string(),
            6,
        );
        // Each Chinese char is 3 bytes, "你好世界" = 12 bytes
        let text = "你好世界";
        let truncated = driver.truncate_body(text);
        assert!(truncated.contains("[truncated"));
    }

    #[test]
    fn test_messages_summary() {
        use openfang_types::message::{Message, Role};
        let messages = vec![
            Message {
                role: Role::System,
                content: openfang_types::message::MessageContent::Text("sys".to_string()),
            },
            Message {
                role: Role::User,
                content: openfang_types::message::MessageContent::Text("hi".to_string()),
            },
            Message {
                role: Role::Assistant,
                content: openfang_types::message::MessageContent::Text("hello".to_string()),
            },
        ];
        let summary = LoggingDriver::messages_summary(&messages);
        assert!(summary.contains("total=3"));
        assert!(summary.contains("user=1"));
        assert!(summary.contains("assistant=1"));
        assert!(summary.contains("system=1"));
    }

    #[tokio::test]
    async fn test_logging_driver_complete() {
        let inner = Arc::new(FakeDriver);
        let driver = LoggingDriver::new(inner, "fake".to_string(), 2000);
        let request = CompletionRequest {
            model: "test-model".to_string(),
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.7,
            system: Some("You are helpful.".to_string()),
            thinking: None,
        };
        let result = driver.complete(request).await;
        assert!(result.is_ok());
        let response = result.unwrap();
        assert_eq!(response.text(), "Hello from fake driver!");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }
}
