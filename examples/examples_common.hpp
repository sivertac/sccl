#pragma once
#ifndef EXAMPLES_COMMON_HEADER
#define EXAMPLES_COMMON_HEADER

#include <binary_util.hpp>
#include <chrono>
#include <optional>
#include <random>
#include <sccl.h>
#include <type_traits>
#include <vulkan/vk_enum_string_helper.h>

#define UNWRAP_VKRESULT(cmd)                                                   \
    do {                                                                       \
        VkResult UNWRAP_VKRESULT_result = cmd;                                 \
        if (UNWRAP_VKRESULT_result != VK_SUCCESS) {                            \
            fprintf(stderr, "Vulkan error: %s\n",                              \
                    string_VkResult(UNWRAP_VKRESULT_result));                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define UNWRAP_SCCL_ERROR(cmd)                                                 \
    do {                                                                       \
        sccl_error_t UNWRAP_SCCL_ERROR_error = cmd;                            \
        if (UNWRAP_SCCL_ERROR_error != sccl_success) {                         \
            fprintf(stderr, "SCCL error: %s\n",                                \
                    sccl_get_error_string(UNWRAP_SCCL_ERROR_error));           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/**
 * @defgroup TimerMacros Timer Macros
 * @{
 */

/**
 * @def START_TIMER(label)
 * @brief Start a timer with a specified label.
 *
 * This macro starts a timer with a given label using the current
 * high-resolution clock.
 *
 * @param label The label for the timer.
 */
#define START_TIMER(label)                                                     \
    std::chrono::high_resolution_clock::time_point START_TIMER_##label =       \
        std::chrono::high_resolution_clock::now();

/**
 * @def STOP_TIMER(label)
 * @brief Stop a timer and print the elapsed time.
 *
 * This macro stops a timer labeled with the specified label, calculates the
 * elapsed time, and prints it in nanoseconds to the standard output.
 *
 * @param label The label of the timer to stop.
 */
#define STOP_TIMER(label)                                                      \
    do {                                                                       \
        std::chrono::high_resolution_clock::duration diff =                    \
            std::chrono::high_resolution_clock::now() - START_TIMER_##label;   \
        printf(#label " time: %lu ns (%lu ms)\n", diff.count(),                \
               std::chrono::duration_cast<std::chrono::milliseconds>(diff)     \
                   .count());                                                  \
    } while (0)

/**
 * @}
 */

void print_data_buffer(const sccl_buffer_t buffer, size_t size);

std::optional<std::string> read_file(const char *filepath);

// template <typename T, typename ...Params>
// constexpr void fill_across_bins(T total, Params& ... params) {
//     static_assert(std::conjunction<std::is_convertible<Params, T>...>::value,
//     "All Params need to be convertible");
// }

/**
 * @brief Assigns a value to a target object, limiting it to a specified maximum
 * size.
 *
 * This function template populates a target object with a value derived from a
 * total value, ensuring that the assigned value does not exceed a specified
 * maximum size. The function enforces type convertibility between the total
 * value and the target object to ensure compatibility.
 *
 * @tparam T The type of the total value.
 * @tparam U The type of the target object.
 * @tparam S The type of the maximum size.
 *
 * @param total The total value to be assigned.
 * @param target The target object to be assigned the value.
 * @param max_size The maximum size to limit the assigned value.
 *
 * @note The function employs a static assertion to verify type convertibility
 * between T and U.
 *
 * @return None.
 *
 * @warning If the target object type is not convertible from the total value
 * type, a static assertion will trigger.
 *
 * @remark If the total value exceeds the maximum size, the target object will
 * be assigned the maximum size; otherwise, it will be assigned the total value.
 *
 * @see std::is_convertible
 */
template <typename T, typename U, typename S>
constexpr void assign_limited(T total, U &target, S max_size)
{
    static_assert(std::is_convertible<U, T>::value,
                  "Params need to be convertible");
    target = (total > max_size) ? max_size : total;
}

/**
 * @brief Fills an array with random values.
 *
 * This function template populates an array with random values of type T.
 *
 * @tparam T The type of values to generate.
 *
 * @param data Pointer to the beginning of the array.
 * @param size The size of the array.
 *
 * @return None.
 */
template <typename T> void fill_array_random(T *data, size_t size)
{
    /* Initialize a random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());

    /* Define the distribution based on the type T */
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<T> distribution(
            std::numeric_limits<T>::min(), std::numeric_limits<T>::max());

        /* Fill the array with random values */
        for (size_t i = 0; i < size; ++i) {
            data[i] = distribution(gen);
        }
    } else if (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> distribution(0.0, 1.0);

        /* Fill the array with random values */
        for (size_t i = 0; i < size; ++i) {
            data[i] = distribution(gen);
        }
    }
}

template <class Container> void print_container(const Container &container)
{
    for (auto it = std::begin(container); it != std::end(container);
         std::advance(it, 1)) {
        std::printf("%s, ", std::to_string(*it).c_str());
    }
    std::printf("\n");
}

template <typename T>
bool float_equal(T a, T b, T epsilon = std::numeric_limits<T>::epsilon())
{
    T abs_th = std::numeric_limits<T>::min();
    T diff = std::abs(a - b);
    T norm =
        std::min((std::fabs(a) + std::fabs(b)), std::numeric_limits<T>::max());
    return diff < std::max(abs_th, epsilon * norm);
}

#endif // EXAMPLES_COMMON_HEADER
