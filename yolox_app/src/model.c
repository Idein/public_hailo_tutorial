#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "hailo/hailort.h"

#define MAX_EDGE_LAYERS (16)
#define HEF_FILE ("yolox_tiny.hef")

static hailo_vdevice vdevice = NULL;
static hailo_hef hef = NULL;
static hailo_configure_params_t config_params = {0};
static hailo_configured_network_group network_group = NULL;
static size_t network_group_size = 1;
static hailo_input_vstream_params_by_name_t input_vstream_params[MAX_EDGE_LAYERS] = {0};
static hailo_output_vstream_params_by_name_t output_vstream_params[MAX_EDGE_LAYERS] = {0};
static hailo_activated_network_group activated_network_group = NULL;
static hailo_vstream_info_t output_vstreams_info[MAX_EDGE_LAYERS] = {0};
static hailo_input_vstream input_vstreams[MAX_EDGE_LAYERS] = {NULL};
static hailo_output_vstream output_vstreams[MAX_EDGE_LAYERS] = {NULL};
static size_t input_vstreams_size = MAX_EDGE_LAYERS;
static size_t output_vstreams_size = MAX_EDGE_LAYERS;

#define READ_AND_DEQUANTIZE(idx, type, out, size) \
    do { \
        type buf[size]; \
        hailo_status status = HAILO_UNINITIALIZED; \
        status = hailo_vstream_read_raw_buffer(output_vstreams[idx], buf, size * sizeof(type)); \
        assert(status == HAILO_SUCCESS); \
        float scale = output_vstreams_info[idx].quant_info.qp_scale; \
        float zp = output_vstreams_info[idx].quant_info.qp_zp; \
        for (int i = 0; i < size; i++) \
            *out++ = scale * (buf[i] - zp); \
    } while (0)

void infer(
    unsigned char *input0,
    float *out0,
    float *out1,
    float *out2)
{
    hailo_status status = HAILO_UNINITIALIZED;
    status = hailo_vstream_write_raw_buffer(input_vstreams[0], input0, 416 * 416 * 3);
    assert(status == HAILO_SUCCESS);
    status = hailo_flush_input_vstream(input_vstreams[0]);
    assert(status == HAILO_SUCCESS);

    READ_AND_DEQUANTIZE(0, uint8_t, out0, 52*52*85);
    READ_AND_DEQUANTIZE(1, uint8_t, out1, 26*26*85);
    READ_AND_DEQUANTIZE(2, uint8_t, out2, 13*13*85);
}

int init()
{
    hailo_status status = HAILO_UNINITIALIZED;

    status = hailo_create_vdevice(NULL, &vdevice);
    assert(status == HAILO_SUCCESS);

    status = hailo_create_hef_file(&hef, HEF_FILE);
    assert(status == HAILO_SUCCESS);

    status = hailo_init_configure_params(hef, HAILO_STREAM_INTERFACE_PCIE, &config_params);
    assert(status == HAILO_SUCCESS);

    status = hailo_configure_vdevice(vdevice, hef, &config_params, &network_group, &network_group_size);
    assert(status == HAILO_SUCCESS);

    status = hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
        input_vstream_params, &input_vstreams_size);
    assert(status == HAILO_SUCCESS);

    status = hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO,
        output_vstream_params, &output_vstreams_size);
    assert(status == HAILO_SUCCESS);

    status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstreams_size, input_vstreams);
    assert(status == HAILO_SUCCESS);

    status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstreams_size, output_vstreams);
    assert(status == HAILO_SUCCESS);

    status = hailo_activate_network_group(network_group, NULL, &activated_network_group);
    assert(status == HAILO_SUCCESS);

    for (size_t i = 0; i < output_vstreams_size; i++)
        hailo_get_output_vstream_info(output_vstreams[i], &output_vstreams_info[i]);

    // for (size_t i = 0; i < input_vstreams_size; i++) {
    //     hailo_vstream_info_t input_vstreams_info;
    //     hailo_get_input_vstream_info(input_vstreams[i], &input_vstreams_info);
    //     printf("==== input[%d] ====\n", i);
    //     printf("direction: %d\n", input_vstreams_info.direction);
    //     printf("format.type: %d\n", input_vstreams_info.format.type);
    //     printf("format.order: %d\n", input_vstreams_info.format.order);
    //     printf("format.flags: %d\n", input_vstreams_info.format.flags);
    //     printf("height: %d\n", input_vstreams_info.shape.height);
    //     printf("width: %d\n", input_vstreams_info.shape.width);
    //     printf("features: %d\n", input_vstreams_info.shape.features);
    //     printf("qp_zp: %f\n", input_vstreams_info.quant_info.qp_zp);
    //     printf("qp_scale: %f\n", input_vstreams_info.quant_info.qp_scale);
    // }
    // printf("========\n");

    // for (size_t i = 0; i < output_vstreams_size; i++) {
    //     printf("==== output[%d] ====\n", i);
    //     printf("direction: %d\n", output_vstreams_info[i].direction);
    //     printf("format.type: %d\n", output_vstreams_info[i].format.type);
    //     printf("format.order: %d\n", output_vstreams_info[i].format.order);
    //     printf("format.flags: %d\n", output_vstreams_info[i].format.flags);
    //     printf("height: %d\n", output_vstreams_info[i].shape.height);
    //     printf("width: %d\n", output_vstreams_info[i].shape.width);
    //     printf("features: %d\n", output_vstreams_info[i].shape.features);
    //     printf("qp_zp: %f\n", output_vstreams_info[i].quant_info.qp_zp);
    //     printf("qp_scale: %f\n", output_vstreams_info[i].quant_info.qp_scale);
    // }
    // printf("========\n");
    // fflush(stdout);

    return status;
}

void destroy()
{
    (void)hailo_deactivate_network_group(activated_network_group);
    (void)hailo_release_output_vstreams(output_vstreams, output_vstreams_size);
    (void)hailo_release_input_vstreams(input_vstreams, input_vstreams_size);
    (void)hailo_release_hef(hef);
    (void)hailo_release_vdevice(vdevice);
}