#include <unistd.h>
#include <pthread.h>
#include "core/graph.hpp"


#define MAX_DEPTH 100000000

int main(int argc, char ** argv)
{
    if (argc<3)
    {
        fprintf(stderr, "usage: heter [path] [jobs] [iterations] [cache size in MB] [graph size in MB] [memory budget in GB]\n");
        exit(-1);
    }
    std::string path = argv[1];
    int PRO_NUM = atoi(argv[2]);
    int TOT_NUM = 8;
    int curr_job_num = 1;
    int curr_bfs = 1;
    int curr_cc = 0;
    int curr_pr = 0;
    int curr_sssp = 0;
    int iterations = atoi(argv[3]);
    // VertexId start_vid = atoi(argv[4]);
    long cache_size = atol(argv[4])*1024l*1024l;
    long graph_size = atol(argv[5])*1024l*1024l;
    long memory_bytes = (argc>=7)?atol(argv[6])*1024l*1024l*1024l:8l*1024l*1024l*1024l;

    // int parallelism = std::thread::hardware_concurrency();
    // printf("parallelism: %d\n",parallelism);
    // if(parallelism>PRO_NUM)
    //     parallelism=PRO_NUM;
    int parallelism = PRO_NUM;
    float arrival_time[8] = {0};
    for (int i = 1; i < 8; ++i)
        arrival_time[i] = atof(argv[6+i]);

    double begin_time = get_time();
    Graph graph(path);
    graph.set_memory_bytes(memory_bytes);

    BigVector<float> pagerank[PRO_NUM];
    BigVector<float> sum[PRO_NUM];
    BigVector<VertexId> degree(graph.path+"/degree", graph.vertices);
    degree.fill(0);

    long vertex_data_bytes = (long)graph.vertices * ( sizeof(VertexId)+ sizeof(float) + sizeof(float));
    graph.set_vertex_data_bytes(vertex_data_bytes);
    printf("Vertices: %d Edges: %ld\n", graph.vertices, graph.edges);

    VertexId active_vertices = 8;
    //sssp
    Bitmap * active_in_sssp[PRO_NUM];
    Bitmap * active_out_sssp[PRO_NUM];
    BigVector<VertexId> depth[PRO_NUM];

    //bfs
    Bitmap * active_in_bfs[PRO_NUM];
    Bitmap * active_out_bfs[PRO_NUM];
    BigVector<VertexId> parent[PRO_NUM];

    //wcc
    Bitmap * active_in_wcc[PRO_NUM];
    Bitmap * active_out_wcc[PRO_NUM];
    BigVector<VertexId> label[PRO_NUM];

    #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
    for(int i = 0; i< PRO_NUM; i++){
        pagerank[i].init(graph.path+"/pagerank"+std::to_string(i), graph.vertices);
        sum[i].init(graph.path+"/sum"+std::to_string(i), graph.vertices);

        active_in_sssp[i] = graph.alloc_bitmap();
        active_out_sssp[i] = graph.alloc_bitmap();
        depth[i].init(graph.path+"/depth"+std::to_string(i), graph.vertices);
        active_out_sssp[i]->clear();
        int start_sssp = 101 * (i+1) + 1;
        if (start_sssp >= graph.vertices)
            start_sssp = (i + 1) * 2;
        active_out_sssp[i]->set_bit(start_sssp);
        depth[i].fill(MAX_DEPTH);
        depth[i][start_sssp] = 0;
        // active_out_sssp[i]->fill();

        active_in_bfs[i] = graph.alloc_bitmap();
        active_out_bfs[i] = graph.alloc_bitmap();
        parent[i].init(graph.path+"/parent"+std::to_string(i), graph.vertices);
        //graph.set_vertex_data_bytes( graph.vertices * sizeof(VertexId) );
        int start_bfs = 71 * (i+1) + 2;
        if (start_bfs >= graph.vertices)
            start_bfs = i + 1;
        active_out_bfs[i]->clear();
        active_out_bfs[i]->set_bit(start_bfs);
        parent[i].fill(-1);
        parent[i][start_bfs] = start_bfs;
        // active_out_bfs[i]->fill();

        active_in_wcc[i] = graph.alloc_bitmap();
        active_out_wcc[i] = graph.alloc_bitmap();
        label[i].init(graph.path+"/label"+std::to_string(i), graph.vertices);
        //graph.set_vertex_data_bytes( graph.vertices * sizeof(VertexId));
        active_out_wcc[i]->fill();
    }
    #pragma omp barrier
    double start_time = get_time();

    graph.set_sizeof_blocks(cache_size, graph_size, vertex_data_bytes*4*PRO_NUM);

    graph.get_should_access_shard(graph.should_access_shard, nullptr);
    graph.stream_edges<VertexId>(
        [&](Edge & e){
        write_add(&degree[e.source], 1);

        return 0;
    },[&](Edge & e){return 0;},
    [&](Edge & e){return 0;},
    [&](Edge & e){return 0;}, nullptr, 0, 0
    );
    printf("degree calculation used %.2f seconds\n", get_time() - begin_time);
    fflush(stdout);

    for (int i = 0; i < PRO_NUM; ++i)
        graph.hint(pagerank[i], sum[i]);
    graph.stream_vertices<VertexId>(
        [&](VertexId i)
    {
        for(int j = 0; j < PRO_NUM; j++){
            //pagerank
            pagerank[j][i] = 1.f / degree[i];
            sum[j][i] = 0;
            //wcc
            label[j][i] = i;
        }
        return 0;
    }, nullptr, 0,
    [&](std::pair<VertexId,VertexId> vid_range)
    {
        for(int i = 0; i < PRO_NUM; i++)
        {
            pagerank[i].load(vid_range.first, vid_range.second);
            sum[i].load(vid_range.first, vid_range.second);
        }
    },
    [&](std::pair<VertexId,VertexId> vid_range)
    {
        for(int i = 0; i < PRO_NUM; i++)
        {
            pagerank[i].save();
            sum[i].save();
        }
    }
    );

    bool flag = false;
    int iter_pr = 1000000; // a large number
    for (int iter=0; iter<iter_pr+iterations || active_vertices!=0 | curr_job_num < TOT_NUM; iter++){
        for (int i = curr_job_num; i < TOT_NUM; ++i) {// dynamically add jobs
            if ((double)arrival_time[i] < (double)(get_time() - start_time)) {
                // BFS, CC, PageRank, SSSP
                curr_job_num++;
                if (curr_job_num % 4 == 1){
                    curr_bfs++;
                    active_vertices += 1;
                } else if (curr_job_num % 4 == 2){
                    curr_cc++;
                    active_vertices = graph.vertices;
                } else if (curr_job_num % 4 == 3){
                    curr_pr++;
                    active_vertices = graph.vertices;
                } else {
                    curr_sssp++;
                    active_vertices += 1;
                }
                printf("Add job %d at %f (Iter: %d)\n", curr_job_num, (double)(get_time() - start_time), iter);
            }
        }
        // printf("%7d (%d): %d\n", iter, curr_job_num, active_vertices);
        parallelism = curr_job_num;
        for (int i = 0; i < curr_bfs; ++i)
            graph.hint(parent[i]);

        if(active_vertices!=0){
            graph.clear_should_access_shard(graph.should_access_shard_sssp);
            graph.clear_should_access_shard(graph.should_access_shard_bfs);
            graph.clear_should_access_shard(graph.should_access_shard_wcc);

            #pragma omp parallel for schedule(dynamic) num_threads(curr_sssp)
            for (int i = 0; i < curr_sssp; ++i){
                std::swap(active_in_sssp[i], active_out_sssp[i]);
                active_out_sssp[i]->clear();
                graph.get_should_access_shard(graph.should_access_shard_sssp, active_in_sssp[i]);
            }
            #pragma omp barrier

            #pragma omp parallel for schedule(dynamic) num_threads(curr_bfs)
            for (int i = 0; i < curr_bfs; ++i){
                std::swap(active_in_bfs[i], active_out_bfs[i]);
                active_out_bfs[i]->clear();
                graph.get_should_access_shard(graph.should_access_shard_bfs, active_in_bfs[i]);
            }
            #pragma omp barrier

            #pragma omp parallel for schedule(dynamic) num_threads(curr_cc)
            for (int i = 0; i < curr_cc; ++i){
                std::swap(active_in_wcc[i], active_out_wcc[i]);
                active_out_wcc[i]->clear();
                graph.get_should_access_shard(graph.should_access_shard_wcc, active_in_wcc[i]);
            }
            #pragma omp barrier

            // pr
            if (curr_job_num > 4)
                graph.get_should_access_shard(graph.should_access_shard_pagerank, nullptr);
            if (curr_pr == 2 && !flag) {
                flag = true;
                iter_pr = iter;
                printf("final iter: %d/%d",iter_pr,iter_pr+iterations);
            }
        }

        graph.get_global_should_access_shard(graph.should_access_shard_wcc, graph.should_access_shard_pagerank,
                                             graph.should_access_shard_bfs,graph.should_access_shard_sssp);
        active_vertices = graph.stream_edges<VertexId>([&](Edge & e){
            //pagerank
            for(int i = 0; i < curr_pr; i++){
                write_add(&sum[i][e.target], pagerank[i][e.source]);
            }
            return 0;
        }, [&](Edge & e){
            //SSSP
            int return_state = 0;
            for(int i = 0; i < curr_sssp; i++){
                if (active_in_sssp[i]->get_bit(e.source))
                {
                    int r = depth[i][e.target];
                    int n = depth[i][e.source] + e.weight;
                    if (n < r)
                    {
                        if (cas(&depth[i][e.target], r, n))
                        {
                            active_out_sssp[i]->set_bit(e.target);
                            return_state = 1;
                        }
                    }
                }
            }
            return return_state;
        },[&](Edge & e){
            //bfs
            int return_state = 0;
            for(int i = 0; i < curr_bfs; i++){
                if (active_in_bfs[i]->get_bit(e.source))
                {
                    if (parent[i][e.target] == -1)
                    {
                        if (cas(&parent[i][e.target], -1, e.source))
                        {
                            active_out_bfs[i]->set_bit(e.target);
                            return_state = 1;
                        }
                    }
                }
            }
            return return_state;
        }, [&](Edge & e){
            //wcc
            int return_state = 0;
            for(int i = 0; i < curr_cc; i++){
                if(active_in_wcc[i] ->get_bit(e.source)){
                    if (label[i][e.source]<label[i][e.target]){
                        if (write_min(&label[i][e.target], label[i][e.source])){
                            active_out_wcc[i]->set_bit(e.target);
                            return_state = 1;
                        }
                    }
                }
            }
            return return_state;
        }, nullptr, 0, 1
        // [&](std::pair<VertexId,VertexId> source_vid_range)
        // {
        //     for(int i = 0; i < PRO_NUM; i++)
        //         pagerank[i].lock(source_vid_range.first, source_vid_range.second);
        // },
        // [&](std::pair<VertexId,VertexId> source_vid_range)
        // {
        //     for(int i = 0; i < PRO_NUM; i++)
        //         pagerank[i].unlock(source_vid_range.first, source_vid_range.second);
        // }
        );

        for (int i = 0; i < curr_pr; ++i)
            graph.hint(pagerank[i], sum[i]);
        if (iter==iterations-1)
        {
            graph.stream_vertices<VertexId>(
                [&](VertexId i)
            {
                for(int j = 0; j < curr_pr; j++){
                    pagerank[j][i] = 0.15f + 0.85f * sum[j][i];
                }
                return 0;
            }, nullptr, 0,
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < curr_pr; j++)
                {
                    pagerank[j].load(vid_range.first, vid_range.second);
                }
            },
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < curr_pr; j++)
                {
                    pagerank[j].save();
                }
            }
            );
        }
        else
        {
            graph.stream_vertices<float>(
                [&](VertexId i)
            {
                for(int j = 0; j < curr_pr; j++){
                    pagerank[j][i] = (0.15f + 0.85f * sum[j][i]) / degree[i];
                    sum[j][i] = 0;
                }
                return 0;
            }, nullptr, 0,
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < curr_pr; j++){
                    pagerank[j].load(vid_range.first, vid_range.second);
                    sum[j].load(vid_range.first, vid_range.second);
                }
            },
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < curr_pr; j++){
                    pagerank[j].save();
                    sum[j].save();
                }
            }
            );
        }
    }
    double end_time = get_time();

    #pragma omp parallel for schedule(dynamic) num_threads(curr_cc)
    for (int j = 0; j < curr_cc; ++j)
    {
        BigVector<VertexId> label_stat(graph.path + "/label_stat", graph.vertices);
        label_stat.fill(0);
        graph.stream_vertices<VertexId>([&](VertexId i) {
            write_add(&label_stat[label[j][i]], 1);
            return 1;
        });
        VertexId components = graph.stream_vertices<VertexId>([&](VertexId i) {
            return label_stat[i] != 0;
        });
        printf("CC: %d\n", components);
    }

    for (int j = 0; j < curr_bfs; ++j)
    {
        printf("Len: %d\n", depth[j][0]);
    }

    printf("%d iterations of concurrent jobs (heter) took %.2f seconds\n", iterations, end_time - begin_time);
}