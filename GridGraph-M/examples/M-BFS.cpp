#include <unistd.h>
#include <pthread.h>
#include "core/graph.hpp"
#include <vector>

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
    int iterations = atoi(argv[3]);
    // VertexId start_vid = atoi(argv[4]);
    long cache_size = atol(argv[4])*1024l*1024l;
    long graph_size = atol(argv[5])*1024l*1024l;
    long memory_bytes = (argc>=7)?atol(argv[6])*1024l*1024l*1024l:8l*1024l*1024l*1024l;

    int parallelism = std::thread::hardware_concurrency();
    printf("parallelism: %d\n",parallelism);
    if(parallelism>PRO_NUM)
        parallelism=PRO_NUM;

    double begin_time = get_time();
    Graph graph(path);
    graph.set_memory_bytes(memory_bytes);
    printf("Vertices: %d Edges: %ld\n",graph.vertices,graph.edges);

    long vertex_data_bytes = (long)graph.vertices * (sizeof(VertexId) + sizeof(float) + sizeof(float));
    graph.set_vertex_data_bytes(vertex_data_bytes);
    VertexId active_vertices = PRO_NUM;

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
        // active_out_bfs[i]->fill();
    }
    #pragma omp barrier

    graph.set_sizeof_blocks(cache_size, graph_size, vertex_data_bytes*4*PRO_NUM);

    graph.get_should_access_shard(graph.should_access_shard, nullptr);

    for (int iter = 0; active_vertices != 0; iter++)
    {
        printf("%7d: %d\n", iter, active_vertices);
        for (int i = 0; i < PRO_NUM; i++)
            graph.hint(parent[i]);

        if(active_vertices!=0){
            graph.clear_should_access_shard(graph.should_access_shard_bfs);

            for (int i = 0; i < PRO_NUM; ++i){
                std::swap(active_in_bfs[i], active_out_bfs[i]);
                active_out_bfs[i]->clear();
                graph.get_should_access_shard(graph.should_access_shard_bfs, active_in_bfs[i]);
            }
        }

#ifdef DEBUG
        for (int i = 0; i < PRO_NUM; ++i)
            active_in_bfs[i]->print(10);
#endif

        std::vector<Edge> edges;
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
            for(int i = 0; i < PRO_NUM; i++){
                if(active_in_bfs[i] ->get_bit(e.source)){
                    if (parent[i][e.target]==-1){
                        if (cas(&parent[i][e.target], -1, e.source)){
                            active_out_bfs[i]->set_bit(e.target);
                            bool flag = false;
                            for (int j = 0; j < PRO_NUM; j++)
                                if (i != j && active_out_bfs[j]->get_bit(e.target))
                                    flag = true;
                            if (!flag)
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
    }
    double end_time = get_time();

    printf("%d iterations of concurrent jobs (m-bfs) took %.2f seconds\n", iterations, end_time - begin_time);
}