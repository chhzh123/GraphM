#include <unistd.h>
#include <pthread.h>
#include "core/graph.hpp"

#define MAX_DEPTH 100000000

int main(int argc, char ** argv)
{
    if (argc<3)
    {
        fprintf(stderr, "usage: m-bfs [path] [jobs] [iterations] [cache size in MB] [graph size in MB] [memory budget in GB]\n");
        // bfs 91*(i+1)
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

    // int parallelism = (argc>=8) ? atol(argv[7]) : std::thread::hardware_concurrency();
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
    printf("Vertices: %d Edges: %ld\n",graph.vertices,graph.edges);

    long vertex_data_bytes = (long)graph.vertices * (sizeof(VertexId) + sizeof(float) + sizeof(float));
    graph.set_vertex_data_bytes(vertex_data_bytes);
    VertexId active_vertices = curr_job_num;

    //bfs
    Bitmap * active_in_bfs[PRO_NUM];
    Bitmap * active_out_bfs[PRO_NUM];
    BigVector<VertexId> parent[PRO_NUM];

    #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
    for(int i = 0; i< PRO_NUM; i++){
        active_in_bfs[i] = graph.alloc_bitmap();
        active_out_bfs[i] = graph.alloc_bitmap();
        parent[i].init(graph.path+"/parent"+std::to_string(i), graph.vertices);
        //graph.set_vertex_data_bytes( graph.vertices * sizeof(VertexId) );
        int start_bfs = 91 * (i+1);
        if (start_bfs >= graph.vertices)
            start_bfs = i + 1;
        active_out_bfs[i]->clear();
        active_out_bfs[i]->set_bit(start_bfs);
        parent[i].fill(-1);
        parent[i][start_bfs] = start_bfs;
        // active_out_bfs[i]->fill(); // ?
    }
    #pragma omp barrier

    graph.set_sizeof_blocks(cache_size, graph_size, vertex_data_bytes*4*PRO_NUM);

    graph.get_should_access_shard(graph.should_access_shard, nullptr);

    for (int iter = 0; active_vertices != 0 || curr_job_num < 8; iter++)
    {
        /* The active_vertices is NOT the precise value here.
         * Consider edge 1->2 (BFS1), 3->2 (BFS2), where
         * in one iteration, vertex 2 will be updated by different jobs
         * and will be activated twice.
         * Actually it is just recounted, but the final result is correct.
         */
        for (int i = curr_job_num; i < PRO_NUM; ++i) {// dynamically add jobs
            if ((double)arrival_time[i] < (double)(get_time() - begin_time)) {
                curr_job_num++;
                active_vertices += 1; // remember!
                printf("Add job %d at %f (Iter: %d)\n", curr_job_num, (double)(get_time() - begin_time), iter);
            }
        }
        printf("%7d (%d): %d\n", iter, curr_job_num, active_vertices);
        parallelism = curr_job_num;
        for (int i = 0; i < curr_job_num; i++) {
            graph.hint(parent[i]);
        }

        if(active_vertices!=0){
            graph.clear_should_access_shard(graph.should_access_shard_bfs);

            #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
            for (int i = 0; i < curr_job_num; ++i){
                std::swap(active_in_bfs[i], active_out_bfs[i]);
                active_out_bfs[i]->clear();
                graph.get_should_access_shard(graph.should_access_shard_bfs, active_in_bfs[i]);
            }
            #pragma omp barrier
        }

#ifdef DEBUG
        for (int i = 0; i < PRO_NUM; ++i)
            active_in_bfs[i]->print(10);
#endif

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
            for(int i = 0; i < curr_job_num; i++){
                if(active_in_bfs[i] ->get_bit(e.source)){
                    if (parent[i][e.target]==-1){
                        if (cas(&parent[i][e.target], -1, e.source)){
                            active_out_bfs[i]->set_bit(e.target);
                            return_state = 1;
                        }
                    }
                }
            }
            return return_state;
        }, [&](Edge & e){
            //wcc
            return 0;
        }, nullptr, 0, 1
        );
    }
    double end_time = get_time();

    printf("%d iterations of concurrent jobs (m-bfs) took %.2f seconds\n", iterations, end_time - begin_time);
}