// SPDX-License-Identifier: Apache-2.0
// Log_internal.h — internal logging entry points + MFLAT_LOG_* macros. Private
// (not installed). Pure C++17; safe to include from any .mm or .cpp TU.
#pragma once

#include "metalflat/Log.h"

namespace mflat {
namespace detail {

bool logEnabled(LogLevel level);                  // cheap pre-eval gate
void logMessage(LogLevel level, const char* msg); // already-formatted message
void logf(LogLevel level, const char* fmt, ...)   // printf-style; -Wformat at sites
    __attribute__((format(printf, 2, 3)));        // free fn: fmt=arg2, varargs=arg3

}  // namespace detail
}  // namespace mflat

// Gate BEFORE argument evaluation so vsnprintf / -[NSError localizedDescription]
// are skipped when the level is filtered out.
#define MFLAT_LOG_ERROR(...) do { if (::mflat::detail::logEnabled(::mflat::LogLevel::Error)) \
    ::mflat::detail::logf(::mflat::LogLevel::Error, __VA_ARGS__); } while (0)
#define MFLAT_LOG_WARN(...)  do { if (::mflat::detail::logEnabled(::mflat::LogLevel::Warn )) \
    ::mflat::detail::logf(::mflat::LogLevel::Warn,  __VA_ARGS__); } while (0)
#define MFLAT_LOG_INFO(...)  do { if (::mflat::detail::logEnabled(::mflat::LogLevel::Info )) \
    ::mflat::detail::logf(::mflat::LogLevel::Info,  __VA_ARGS__); } while (0)
