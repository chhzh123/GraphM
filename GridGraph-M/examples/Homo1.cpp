#include <unistd.h>
#include <pthread.h>
#include "core/graph.hpp"


#define MAX_DEPTH 100000000

int main(int argc, char ** argv)
{
    if (argc<3)
    {
        fprintf(stderr, "usage: homo1 [path] [jobs] [iterations] [cache size in MB] [graph size in MB] [memory budget in GB]\n");
        // BFS 10 20 30 40
        // CC *4
        exit(-1);
    }
    std::string path = argv[1];
    int PRO_NUM = atoi(argv[2]);
    int TOT_NUM = 8;
    int curr_job_num = 1;
    int curr_bfs = 1;
    int curr_cc = 0;
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
    printf("Vertices: %d Edges: %ld\n", graph.vertices, graph.edges);

    long vertex_data_bytes = (long)graph.vertices * ( sizeof(VertexId)+ sizeof(float) + sizeof(float));
    graph.set_vertex_data_bytes(vertex_data_bytes);

    VertexId active_vertices = 1;

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

        active_in_bfs[i] = graph.alloc_bitmap();
        active_out_bfs[i] = graph.alloc_bitmap();
        parent[i].init(graph.path+"/parent"+std::to_string(i), graph.vertices);
        //graph.set_vertex_data_bytes( graph.vertices * sizeof(VertexId) );
        int start_bfs = 10 * (i+1);
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

    graph.stream_vertices<VertexId>(
        [&](VertexId i)
    {
        for(int j = 0; j < PRO_NUM; j++){
            //wcc
            label[j][i] = i;
        }
        return 0;
    }, nullptr, 0,
    [&](std::pair<VertexId,VertexId> vid_range)
    {
    },
    [&](std::pair<VertexId,VertexId> vid_range)
    {
    }
    );

    for (int iter=0; active_vertices!=0 || curr_job_num < TOT_NUM; iter++){
        for (int i = curr_job_num; i < TOT_NUM; ++i) {// dynamically add jobs
            if ((double)arrival_time[i] < (double)(get_time() - start_time)) {
                curr_job_num++;
                if (curr_job_num > 4)
                    curr_cc++;
                else
                    curr_bfs++;
                // first add BFS, later CC
                active_vertices = (curr_job_num >= 4 ? graph.vertices : 1 + active_vertices); // remember!
                printf("Add job %d at %f (Iter: %d)\n", curr_job_num, (double)(get_time() - start_time), iter);
            }
        }
        // printf("%7d (%d): %d\n", iter, curr_job_num, active_vertices);
        parallelism = curr_job_num;
        for (int i = 0; i < std::min(curr_job_num,4); i++)
            graph.hint(parent[i]);

        if(active_vertices!=0){
            graph.clear_should_access_shard(graph.should_access_shard_bfs);
            graph.clear_should_access_shard(graph.should_access_shard_wcc);

            #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
            for(int i = 0; i < curr_job_num; i++){
                if (i < 4) {
                    std::swap(active_in_bfs[i], active_out_bfs[i]);
                    active_out_bfs[i]->clear();
                    graph.get_should_access_shard(graph.should_access_shard_bfs, active_in_bfs[i]);
                } else {
                    std::swap(active_in_wcc[i%4], active_out_wcc[i%4]);
                    active_out_wcc[i%4]->clear();
                    graph.get_should_access_shard(graph.should_access_shard_wcc, active_in_wcc[i%4]);
                }
            }
            #pragma omp barrier
        }

        graph.get_global_should_access_shard(graph.should_access_shard_wcc, graph.should_access_shard_pagerank,
                                             graph.should_access_shard_bfs,graph.should_access_shard_sssp);
        active_vertices = graph.stream_edges<VertexId>([&](Edge & e){
            //pagerank
            return 0;
        }, [&](Edge & e){
            //SSSP
            return 0;
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
        );

    }
    double end_time = get_time();

    parallelism = curr_cc;
    #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
    for (int j = 0; j < curr_cc; ++j){
        BigVector<VertexId> label_stat(graph.path + "/label_stat", graph.vertices);
        label_stat.fill(0);
        graph.stream_vertices<VertexId>([&](VertexId i) {
            write_add(&label_stat[label[j][i]], 1);
            return 1;
        });
        VertexId components = graph.stream_vertices<VertexId>([&](VertexId i) {
            return label_stat[i] != 0;
        });
        printf("CC: %d\n",components);
    }

    printf("%d iterations of concurrent jobs (homo1) took %.2f seconds\n", iterations, end_time - begin_time);
}