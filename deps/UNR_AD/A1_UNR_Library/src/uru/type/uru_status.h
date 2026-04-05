/**
 * @file uru_status.h
 * @author Guangnan Feng (nobugday@gamil.com)
 * @brief
 * @version 1.0
 * @date 2023-12-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#define URU_ANY_SOURCE (-6)

#define URU_STATUS_PACK(func) func(URU_STATUS_OK, "Success"),                                                                        \
                              func(URU_STATUS_FUNC_NOT_IMPLEMENTED, "A function is called by a pointer, but it is NOT implemented"), \
                              func(URU_STATUS_ERROR, "ERROR"),                                                                       \
                              func(URU_CHANNEL_OPENED, "This channel is already opened"),                                            \
                              func(URU_SIGNAL_TRIGGERED, "Signal is triggered"),                                                     \
                              func(URU_SIGNAL_WAITING, "Signal is NOT triggered"),                                                   \
                              func(URU_DATA_RECEIVED, "Data is received"),                                                           \
                              func(URU_DATA_RECEIVING, "Data is NOT received"),                                                      \
                              func(URU_DATA_SENT, "Data is sent"),                                                                   \
                              func(URU_DATA_SENDING, "Data is NOT sent")

#define URU_STATUS_PACK_SELECT_ENUM(STATUS_ENUM, STATUS_DESC) STATUS_ENUM
#define URU_STATUS_PACK_SELECT_DESC(STATUS_ENUM, STATUS_DESC) STATUS_DESC

typedef enum {
    URU_STATUS_PACK(URU_STATUS_PACK_SELECT_ENUM),
    URU_STATUS_NUM
} uru_status_t;

const char* uru_status_string(uru_status_t status);

typedef enum {
    UNR_STATUS_OK = 0,
    URU_ERROR_EXIT,
    URU_NOT_IMPLEMENT_EXIT,
    URU_ASSERT_EXIT,
    URU_GLEX_ERROR_EXIT
} uru_exit_t;