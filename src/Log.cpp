// SPDX-License-Identifier: Apache-2.0
// Log.cpp — the single owner of the process-global log sink + threshold.
// Pure C++ (no Metal/ObjC): messages cross the boundary as bytes only.
#include "metalflat/Log.h"
#include "Log_internal.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>

namespace mflat {
namespace {

void defaultSink(LogLevel lvl, const char* msg, void*) {
    const char* tag = lvl == LogLevel::Error ? "error"
                    : lvl == LogLevel::Warn  ? "warn" : "info";
    std::fprintf(stderr, "[metalflat] %s: %s\n", tag, msg);  // prefix lives HERE only
}

// Constant-initialized => no static ctor / no atexit dtor: safe even if a log
// fires during static init or teardown. nullptr handler == built-in sink.
std::atomic<LogHandler> gHandler{nullptr};
std::atomic<void*>      gUser{nullptr};
std::atomic<int>        gThreshold{static_cast<int>(LogLevel::Info)};

}  // namespace

void setLogHandler(LogHandler h, void* user) {
    gUser.store(user, std::memory_order_release);
    gHandler.store(h, std::memory_order_release);   // publish handler last
}
void     setLogLevel(LogLevel t) { gThreshold.store(static_cast<int>(t), std::memory_order_relaxed); }
LogLevel logLevel()              { return static_cast<LogLevel>(gThreshold.load(std::memory_order_relaxed)); }

namespace detail {

bool logEnabled(LogLevel lvl) {
    const int t = gThreshold.load(std::memory_order_relaxed);
    return t != static_cast<int>(LogLevel::Off) && static_cast<int>(lvl) >= t;
}

void logMessage(LogLevel lvl, const char* msg) {
    if (!logEnabled(lvl)) return;
    if (LogHandler h = gHandler.load(std::memory_order_acquire)) {
        h(lvl, msg, gUser.load(std::memory_order_acquire));   // no library lock held
        return;
    }
    defaultSink(lvl, msg, nullptr);
}

void logf(LogLevel lvl, const char* fmt, ...) {
    if (!logEnabled(lvl)) return;                    // also skips formatting when filtered
    char buf[1024];
    va_list ap; va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof buf, fmt, ap);   // used once -> no va_copy
    va_end(ap);
    if (n < 0) return;                               // encoding error: drop
    logMessage(lvl, buf);                            // buf is always NUL-terminated
}

}  // namespace detail
}  // namespace mflat
