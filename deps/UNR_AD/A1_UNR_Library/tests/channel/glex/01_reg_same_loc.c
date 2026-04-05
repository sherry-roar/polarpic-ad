#include <uri/api/uri.h>
#include <uru/dev/trace.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    uri_channel_h* uri_channel_arr;
    uri_channel_h uri_glex_channel;
    int uri_num_channels;
    uri_query_channel_list(&uri_channel_arr, &uri_num_channels);

    int find_channel_glex = 0;

    for (int i = 0; i < uri_num_channels; i++) {
        uri_channel_attr_t* attr;
        uri_channel_attr_get(uri_channel_arr[i], &attr);

        if (strcmp(attr->type, "glex") == 0) {
            find_channel_glex = 1;
            uri_glex_channel = uri_channel_arr[i];
        }
    }

    if (!find_channel_glex) {
        printf("ERROR: No GLEX channel is available!\n");
        return 1;
    }
    printf("Find GLEX channel!\n");

    uri_channel_open(uri_glex_channel);

    for (int i = 1; i <= 4095; i++) {
        uri_mem_h uri_mem;
        void* buf = uru_malloc(60*1024*256);
        uri_mem_reg(uri_glex_channel, buf, 1, &uri_mem, NULL);
        printf("Register %lfGB memory\n", (double)i*60*1024*256/1024/1024/1024);
    }

    uri_channel_close(uri_glex_channel);

    return 0;
}