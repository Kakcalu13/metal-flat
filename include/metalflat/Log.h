// SPDX-License-Identifier: Apache-2.0
// metalflat/Log.h — process-global log sink control. Pure C++17, no Metal/ObjC.
//
// By default the library writes diagnostics to stderr as
//   [metalflat] <level>: <message>
// Consumers who don't want that (GUI apps with no console, servers with
// structured logging, notebooks, Swift routing to os_log) can install their own
// handler or raise the threshold — a library should never force stderr output.
#pragma once

namespace mflat {

// Severity, increasing. `Off` is a threshold-only sentinel (never delivered to a
// handler). Values are ABI-stable and mirror mflat_log_level_t 1:1.
enum class LogLevel : int { Info = 0, Warn = 1, Error = 2, Off = 3 };

// Called SYNCHRONOUSLY on the thread that hit the site. `msg` is NUL-terminated
// UTF-8 with NO "[metalflat]" prefix and NO trailing newline — the sink owns all
// presentation. `user` is round-tripped untouched. The handler must not throw
// and, if the process issues concurrent searches, must be thread-safe.
using LogHandler = void (*)(LogLevel level, const char* msg, void* user);

// Install a sink. handler == nullptr RESTORES the built-in stderr sink (it does
// NOT silence — use setLogLevel(LogLevel::Off) for that). Configure once before
// concurrent use.
void setLogHandler(LogHandler handler, void* user = nullptr);

// Emit only messages with severity >= threshold. Default Info (emit all);
// Off drops everything.
void     setLogLevel(LogLevel threshold);
LogLevel logLevel();

}  // namespace mflat
