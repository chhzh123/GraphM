#include <unistd.h>
#include <pthread.h>
#include "core/graph.hpp"


#define MAX_DEPTH 100000000

int main(int argc, char ** argv)
{
    if (argc<3)
    {
        fprintf(stderr, "usage: m-sssp [path] [jobs] [iterations] [cache size in MB] [graph size in MB] [memory budget in GB]\n");
        exit(-1);
    }
    std::string path = argv[1];
    int PRO_NUM = atoi(argv[2]);
    int curr_job_num = 1;
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
    for (int i = 0; i < 8; ++i)
        arrival_time[i] = i*0.5;

    double begin_time = get_time();
    Graph graph(path);
    graph.set_memory_bytes(memory_bytes);
    printf("Vertices: %d Edges: %ld\n", graph.vertices, graph.edges);

    long vertex_data_bytes = (long)graph.vertices * ( sizeof(VertexId)+ sizeof(float) + sizeof(float));
    graph.set_vertex_data_bytes(vertex_data_bytes);
    VertexId active_vertices = curr_job_num;

    //sssp
    Bitmap * active_in_sssp[PRO_NUM];
    Bitmap * active_out_sssp[PRO_NUM];
    BigVector<VertexId> depth[PRO_NUM];

    #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
    for(int i = 0; i< PRO_NUM; i++){
        active_in_sssp[i] = graph.alloc_bitmap();
        active_out_sssp[i] = graph.alloc_bitmap();
        depth[i].init(graph.path+"/depth"+std::to_string(i), graph.vertices);
        active_out_sssp[i]->clear();
        int start_sssp = 211 * (i+1);
        if (start_sssp >= graph.vertices)
            start_sssp = i + 1;
        active_out_sssp[i]->set_bit(start_sssp);
        depth[i].fill(MAX_DEPTH);
        depth[i][start_sssp] = 0;
        // active_out_sssp[i]->fill();
    }
    #pragma omp barrier

    graph.set_sizeof_blocks(cache_size, graph_size, vertex_data_bytes*4*PRO_NUM);

    graph.get_should_access_shard(graph.should_access_shard, nullptr);

    for (int iter=0; active_vertices!=0 || curr_job_num < 8; iter++){
        for (int i = curr_job_num; i < PRO_NUM; ++i) {// dynamically add jobs
            if ((double)arrival_time[i] < (double)(get_time() - begin_time)) {
                curr_job_num++;
                active_vertices += 1; // remember!
                printf("Add job %d at %f (Iter: %d)\n", curr_job_num, (double)(get_time() - begin_time), iter);
            }
        }
        printf("%7d (%d): %d\n", iter, curr_job_num, active_vertices);
        parallelism = curr_job_num;

        if(active_vertices!=0){
            graph.clear_should_access_shard(graph.should_access_shard_sssp);

            #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
            for (int i = 0; i < curr_job_num; ++i){
                std::swap(active_in_sssp[i], active_out_sssp[i]);
                active_out_sssp[i]->clear();
                graph.get_should_access_shard(graph.should_access_shard_sssp, active_in_sssp[i]);
            }
            #pragma omp barrier
        }

#ifdef DEBUG
        for (int i = 0; i < PRO_NUM; ++i)
            active_in_sssp[i]->print(10);
#endif

        graph.get_global_should_access_shard(graph.should_access_shard_wcc, graph.should_access_shard_pagerank,
                                             graph.should_access_shard_bfs,graph.should_access_shard_sssp);
        active_vertices = graph.stream_edges<VertexId>([&](Edge & e){
            //pagerank
            return 0;
        }, [&](Edge & e){
            //SSSP
            int return_state = 0;
            for(int i = 0; i < curr_job_num; i++){
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
            return 0;
        }, [&](Edge & e){
            //wcc
            return 0;
        }, nullptr, 0, 1
        );
    }
    double end_time = get_time();

    for (int j = 0; j < PRO_NUM; ++j)
    {
        printf("Len: %d\n", depth[j][0]);
    }

    printf("%d iterations of concurrent jobs (m-sssp) took %.2f seconds\n", iterations, end_time - begin_time);
}