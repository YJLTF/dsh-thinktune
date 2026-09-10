/**
 * Shared HTTP transport for both wire protocols: JSON requests with harness
 * attribution headers, credential resolution, status-code → stable LlmError
 * code mapping, and an idle-timeout-guarded line reader over the response
 * body. Caller aborts surface as `ABORTED`; a read stall past
 * `idleTimeoutMs` as `TIMEOUT`.
 */
import { LlmError, assertUsableApiKey, attributionHeaders } from '@deepseek-ai/dsh-llm';
/** Map a non-2xx provider status (plus optional body message) to the shared failure-code taxonomy. */
export function httpErrorCode(status, message) {
    if (status === 401 || status === 403)
        return 'AUTH';
    if (status === 429)
        return 'RATE_LIMIT';
    if (status === 404)
        return 'INVALID_REQUEST';
    if (status === 400 && /exceed/i.test(message) && /context|token|length/i.test(message))
        return 'CONTEXT_WINDOW_EXCEEDED';
    if (status === 400 && /quota/i.test(message))
        return 'QUOTA';
    if (status === 400 || status === 413 || status === 422)
        return 'INVALID_REQUEST';
    if (status >= 500)
        return 'SERVER';
    return `HTTP_${status}`;
}
export function isAbortReason(error) {
    return error instanceof Error && (error.name === 'AbortError' || error.name === 'TimeoutError');
}
/** Resolve the optional bearer credential from the environment reference named in config. */
export function resolveBearer(apiKeyEnv, env) {
    if (!apiKeyEnv)
        return undefined;
    const raw = env[apiKeyEnv];
    if (raw === undefined) {
        throw new LlmError(`thinktune: credential reference "${apiKeyEnv}" is not set in the process environment`, 'MISSING_CREDENTIAL');
    }
    return assertUsableApiKey(raw, 'thinktune-ollama', `process.env.${apiKeyEnv}`);
}
/** Extract a human-readable message from an error body (JSON `error`/`message` or raw text). */
function bodyMessage(text) {
    if (text.length === 0)
        return '';
    try {
        const parsed = JSON.parse(text);
        if (typeof parsed.error === 'string')
            return parsed.error;
        if (parsed.error && typeof parsed.error === 'object' && 'message' in parsed.error) {
            return String(parsed.error.message ?? text);
        }
        if (typeof parsed.message === 'string')
            return parsed.message;
    }
    catch { }
    return text;
}
/** Send one request with attribution and optional bearer headers; non-2xx throws a coded LlmError. */
async function send(url, init, bearer) {
    const headers = {
        ...init.headers,
        ...attributionHeaders(),
    };
    if (bearer)
        headers.authorization = `Bearer ${bearer}`;
    let response;
    try {
        response = await fetch(url, { ...init, headers });
    }
    catch (error) {
        if (isAbortReason(error))
            throw new LlmError('thinktune: provider request aborted', 'ABORTED');
        throw new LlmError(`thinktune: provider request failed: ${error.message}`, 'PROVIDER_UNAVAILABLE');
    }
    if (!response.ok) {
        const message = bodyMessage(await response.text().catch(() => ''));
        throw new LlmError(`thinktune: provider HTTP ${response.status}${message ? `: ${message}` : ''}`, httpErrorCode(response.status, message), { status: response.status });
    }
    return response;
}
/** POST one JSON request and return the raw response; non-2xx throws a coded LlmError. */
export function postJson(options) {
    return send(options.url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(options.body),
        signal: options.signal,
    }, options.bearer);
}
/** GET helper for `/api/tags` with the same failure mapping. */
export function getJson(url, bearer, signal) {
    return send(url, { method: 'GET', signal }, bearer).then((response) => response.json());
}
/**
 * Yield newline-delimited lines from a streaming JSON response body, guarding
 * every read with the idle timeout. The reader is released on completion,
 * caller early-return, stall, and abort alike.
 */
export async function* readLines(response, signal, idleTimeoutMs) {
    if (!response.body)
        throw new LlmError('thinktune: provider response has no body', 'PROVIDER_HTTP_ERROR');
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let timer;
    const clearTimer = () => {
        if (timer !== undefined) {
            clearTimeout(timer);
            timer = undefined;
        }
    };
    let completed = false;
    let stalled = false;
    try {
        while (true) {
            let result;
            if (idleTimeoutMs > 0) {
                clearTimer();
                timer = setTimeout(() => {
                    stalled = true;
                    void reader.cancel().catch(() => { });
                }, idleTimeoutMs);
            }
            try {
                result = await reader.read();
            }
            finally {
                clearTimer();
            }
            if (stalled) {
                throw new LlmError(`thinktune: provider stream idle past ${idleTimeoutMs}ms`, 'TIMEOUT');
            }
            if (result.done)
                break;
            if (signal?.aborted)
                throw new LlmError('thinktune: provider stream aborted', 'ABORTED');
            buffer += decoder.decode(result.value, { stream: true });
            let newline = buffer.indexOf('\n');
            while (newline >= 0) {
                const line = buffer.slice(0, newline).replace(/\r$/, '');
                buffer = buffer.slice(newline + 1);
                if (line.length > 0)
                    yield line;
                newline = buffer.indexOf('\n');
            }
        }
        if (signal?.aborted)
            throw new LlmError('thinktune: provider stream aborted', 'ABORTED');
        const rest = buffer.replace(/\r$/, '');
        if (rest.length > 0)
            yield rest;
        completed = true;
    }
    catch (error) {
        if (signal?.aborted || isAbortReason(error)) {
            throw new LlmError('thinktune: provider stream aborted', 'ABORTED');
        }
        throw error;
    }
    finally {
        clearTimer();
        if (!completed)
            void reader.cancel().catch(() => { });
    }
}
