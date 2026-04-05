#include <uri/verbs/uri_verbs.h>
#include <uru/sys/time.h>

uru_status_t uri_verbs_event_fill(uri_verbs_channel_t* info, verbs_imm_protocol_t* event, int dst_rank, uru_flag_t* flag_p, int64_t add_num) {
    URU_ASSERT_REASON(add_num == -1, "add_num must be `-1` in VERBS channel!");
    URU_ASSERT(0 <= dst_rank && dst_rank < uru_mpi->size);
    URU_ASSERT(flag_pool->rmt_base[dst_rank] <= flag_p && flag_p < flag_pool->rmt_base[dst_rank] + flag_pool->size);
    event->flag_p32 = (u_int32_t)(flag_p - flag_pool->rmt_base[dst_rank]);
    return URU_STATUS_OK;
}

void uri_verbs_event_progress(uri_verbs_channel_t* info, verbs_imm_protocol_t* event) {
    if (event->flag_p32 == 0) {
        return;
    }
    uru_flag_t* flag_p = flag_pool->rmt_base[uru_mpi->rank] + event->flag_p32;
    flag_p->counter += -1;
    if (flag_p->counter == 0 && flag_p->queue) {
        if (flag_p->queue) {
            uru_queue_push(flag_p->queue, flag_p->idx, 1);
        }
    }
}

uru_status_t uri_verbs_rdma_plan_create(
    uri_channel_t* channel_p, uint8_t once_now, size_t num_wrt,
    uri_mem_t** loc_mem_pp, size_t* loc_offset_arr, uru_flag_t** loc_flag_p_arr, int64_t* loc_flag_add_arr, uint64_t* wrt_size_arr,
    uri_mem_t** rmt_mem_pp, size_t* rmt_offset_arr, uru_flag_t** rmt_flag_p_arr, int64_t* rmt_flag_add_arr,
    uri_plan_t** plan_pp) {

    uri_verbs_channel_t* info = (uri_verbs_channel_t*)channel_p->info;

    uri_plan_t* plan_p = (uri_plan_t*)uru_calloc(1, sizeof(uri_plan_t) + sizeof(uri_verbs_plan_t));
    *plan_pp = plan_p;
    uri_verbs_plan_t* verbs_plan_p = (uri_verbs_plan_t*)(plan_p + 1);

    verbs_plan_p->num_wr = num_wrt;
    struct ibv_send_wr* wr = (struct ibv_send_wr*)uru_calloc(num_wrt, sizeof(struct ibv_send_wr));
    struct ibv_sge* sg_list = (struct ibv_sge*)uru_calloc(num_wrt, sizeof(struct ibv_sge));
    verbs_plan_p->dst_rank = (int*)uru_calloc(num_wrt, sizeof(int));
    verbs_plan_p->wr = wr;
    verbs_plan_p->sg_list = sg_list;
    for (int i = 0; i < num_wrt; i++) {
        uri_verbs_mem_t* send_verbs_mem = (uri_verbs_mem_t*)(loc_mem_pp[i] + 1);
        uri_verbs_mem_t* recv_verbs_mem = (uri_verbs_mem_t*)(rmt_mem_pp[i] + 1);
        void* send_buf_addr;
        void* recv_buf_addr;
        size_t send_mem_size, recv_mem_size;
        int src_rank, dst_rank;
        uri_mem_get_info(channel_p, loc_mem_pp[i], &send_buf_addr, &send_mem_size, &src_rank);
        uri_mem_get_info(channel_p, rmt_mem_pp[i], &recv_buf_addr, &recv_mem_size, &dst_rank);

        verbs_plan_p->dst_rank[i] = dst_rank;
        wr[i].next = NULL;
        wr[i].sg_list = sg_list + i;
        wr[i].num_sge = 1;
        wr[i].opcode = IBV_WR_RDMA_WRITE;
        sg_list[i].addr = (uint64_t)(send_buf_addr + loc_offset_arr[i]);
        sg_list[i].length = wrt_size_arr[i];
        sg_list[i].lkey = send_verbs_mem->lkey;
        wr[i].wr.rdma.remote_addr = (uint64_t)(recv_buf_addr + rmt_offset_arr[i]);
        wr[i].wr.rdma.rkey = recv_verbs_mem->rkey;
        if (VERBS_MAX_INLINE_DATA >= wrt_size_arr[i]) {
            wr[i].send_flags |= IBV_SEND_INLINE;
        }

        { /* Set send event: Each send will post an send finish event in SRQ */
            wr[i].send_flags |= IBV_SEND_SIGNALED;
            verbs_wrid_protocol_t *wrid = (verbs_wrid_protocol_t*)(&(wr[i].wr_id));
            wrid->dst_rank = dst_rank;
            if (loc_flag_p_arr[i]) {
                uri_verbs_event_fill(info, (verbs_imm_protocol_t*)(&(wrid->imm_protocol)), src_rank, loc_flag_p_arr[i], loc_flag_add_arr[i]);
            }
        }

        { /* Set recv event: Only message with remote event will post an receive finish event in SRQ */
            if (rmt_flag_p_arr[i]) {
                wr[i].opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
                uri_verbs_event_fill(info, (verbs_imm_protocol_t*)(&(wr[i].imm_data)), dst_rank, rmt_flag_p_arr[i], rmt_flag_add_arr[i]);
            }
        }

    }

    if (once_now) {
        uri_verbs_rdma_plan_start(channel_p, plan_p);
        uri_verbs_rdma_plan_destroy(channel_p, plan_p);
    }

    return URU_STATUS_OK;
}

uru_status_t uri_verbs_rdma_plan_start(uri_channel_t* channel_p, uri_plan_t* plan_p) {

    uri_verbs_channel_t* info = (uri_verbs_channel_t*)channel_p->info;
    uri_verbs_plan_t* verbs_plan_p = (uri_verbs_plan_t*)(plan_p + 1);

    pthread_spin_lock(&(info->send_lock));

    for (int i = 0; i < verbs_plan_p->num_wr; i++) {
        struct ibv_send_wr* wr = verbs_plan_p->wr + i;
        struct ibv_send_wr* bad_wr;
        int dst_rank = verbs_plan_p->dst_rank[i];
        volatile int64_t* req_submit = (volatile int64_t*)(info->req_submit + dst_rank);
        volatile int64_t* req_finish = (volatile int64_t*)(info->req_finish + dst_rank);
        URU_ASSERT((*req_submit) >= (*req_finish));
        for (uru_cpu_cyc_t start = uru_cpu_cyc_now(), now = start;
                ((*req_submit) - (*req_finish)) > (VERBS_OUTSTANDING_SEND_REQUESTS - 1);
                now = uru_cpu_cyc_now()) {
            /* QP sending queue is full. Wait and do nothing */
            URU_ASSERT((*req_submit) >= (*req_finish));
            if (uru_time_diff(start, now) > 10) {
                printf("Waiting for space in qp[%d] Timeout!\n", dst_rank);
                exit(URU_ERROR_EXIT);
            }
        }
        verbs_check(ibv_post_send(info->qp_list[dst_rank], wr, &bad_wr));
        info->req_submit[dst_rank]++;
    }

    pthread_spin_unlock(&(info->send_lock));

    return URU_STATUS_OK;
}

uru_status_t uri_verbs_rdma_plan_destroy(uri_channel_t* channel_p, uri_plan_t* plan_p) {

    uri_verbs_channel_t* info = (uri_verbs_channel_t*)channel_p->info;
    uri_verbs_plan_t* verbs_plan_p = (uri_verbs_plan_t*)(plan_p + 1);

    free(verbs_plan_p->wr);
    free(verbs_plan_p->sg_list);
    free(verbs_plan_p->dst_rank);
    free(plan_p);

    return URU_STATUS_OK;
}