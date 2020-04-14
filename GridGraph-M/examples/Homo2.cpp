#include <unistd.h>
#include <pthread.h>
#include "core/graph.hpp"


#define MAX_DEPTH 100000000

int main(int argc, char ** argv)
{
    if (argc<3)
    {
        fprintf(stderr, "usage: homo2 [path] [jobs] [iterations] [cache size in MB] [graph size in MB] [memory budget in GB]\n");
        // pr *4
        // sssp 71*(i+1) + 2
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

    BigVector<float> pagerank[PRO_NUM];
    BigVector<float> sum[PRO_NUM];
    BigVector<VertexId> degree(graph.path+"/degree", graph.vertices);
    degree.fill(0);

    long vertex_data_bytes = (long)graph.vertices * ( sizeof(VertexId)+ sizeof(float) + sizeof(float));
    graph.set_vertex_data_bytes(vertex_data_bytes);
    printf("Vertices: %d Edges: %ld\n", graph.vertices, graph.edges);

    VertexId active_vertices = 4;
    //sssp
    Bitmap * active_in_sssp[PRO_NUM];
    Bitmap * active_out_sssp[PRO_NUM];
    BigVector<VertexId> depth[PRO_NUM];

    #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
    for(int i = 0; i< PRO_NUM; i++){
        pagerank[i].init(graph.path+"/pagerank"+std::to_string(i), graph.vertices);
        sum[i].init(graph.path+"/sum"+std::to_string(i), graph.vertices);

        active_in_sssp[i] = graph.alloc_bitmap();
        active_out_sssp[i] = graph.alloc_bitmap();
        active_out_sssp[i]->clear();
        int start_sssp = 71 * (i+1) + 2;
        if (start_sssp >= graph.vertices)
            start_sssp = i + 1;
        active_out_sssp[i]->set_bit(start_sssp);
        depth[i].init(graph.path + "/depth" + std::to_string(i), graph.vertices);
        depth[i].fill(MAX_DEPTH);
        depth[i][start_sssp] = 0;
        // active_out_sssp[i]->fill();
    }
    #pragma omp barrier

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


    graph.get_should_access_shard(graph.should_access_shard_pagerank, nullptr);
    for (int iter=0; iter<iterations || active_vertices!=0; iter++){
        printf("%7d: %d\n", iter, active_vertices);

        if(active_vertices!=0){
            graph.clear_should_access_shard(graph.should_access_shard_sssp);

            #pragma omp parallel for schedule(dynamic) num_threads(parallelism)
            for (int i = 0; i < PRO_NUM; ++i){
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
            for(int i = 0; i < PRO_NUM; i++){
                write_add(&sum[i][e.target], pagerank[i][e.source]);
            }
            return 0;
        }, [&](Edge & e){
            //SSSP
            int return_state = 0;
            for(int i = 0; i < PRO_NUM; i++){
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

        for (int i = 0; i < PRO_NUM; ++i)
            graph.hint(pagerank[i], sum[i]);
        if (iter==iterations-1)
        {
            graph.stream_vertices<VertexId>(
                [&](VertexId i)
            {
                for(int j = 0; j < PRO_NUM; j++){
                    pagerank[j][i] = 0.15f + 0.85f * sum[j][i];
                }
                return 0;
            }, nullptr, 0,
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < PRO_NUM; j++)
                {
                    pagerank[j].load(vid_range.first, vid_range.second);
                }
            },
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < PRO_NUM; j++)
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
                for(int j = 0; j < PRO_NUM; j++){
                    pagerank[j][i] = (0.15f + 0.85f * sum[j][i]) / degree[i];
                    sum[j][i] = 0;
                }
                return 0;
            }, nullptr, 0,
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < PRO_NUM; j++){
                    pagerank[j].load(vid_range.first, vid_range.second);
                    sum[j].load(vid_range.first, vid_range.second);
                }
            },
            [&](std::pair<VertexId,VertexId> vid_range)
            {
                for(int j = 0; j < PRO_NUM; j++){
                    pagerank[j].save();
                    sum[j].save();
                }
            }
            );
        }
    }
    double end_time = get_time();

    for(int j = 0; j < PRO_NUM; ++j){
        printf("Len: %d\n",depth[j][0]);
    }

    printf("%d iterations of concurrent jobs (homo2) took %.2f seconds\n", iterations, end_time - begin_time);
}