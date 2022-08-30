#!/usr/bin/env Rscript
library("optparse")
library("docstring")
library("igraph")

form_adjacency = function(fc, remove_diag = TRUE, quantile_threshold = 0.8) {
    #' Form an binary adjacency matrix from a connectivity matrix
    #'
    #' @param fc Symmetric functional connectivity matrix
    #' @param remove_diag Default=TRUE. Remove matrix diagonal (self-correlations)
    #' @param quantile_threshold Default=0.8. Set edges below this quantile to zero.
    #'
    #' @return adj. Binary adjacency matrix
    adj <- as.matrix(fc)
    if(remove_diag) diag(adj) = 0
    adj[adj < quantile(adj, quantile_threshold) ] <- 0
    adj[adj > 0 ] <- 1
	return(adj)
}

graph_metrics = function(adj, session_name, weighted = TRUE, mode = "directed") {
    #' Calculate a set of Graph Theory metrics from an adjacency matrix
    #'
    #' @param adj Adjacency matrix
    #' @param session_name Session ID, e.g. CU0011_baseline
    #' @param weighted Default=TRUE. Calculate metrics assuming weighted edge-to-edge connections
    #' @param mode mode="directed". Can be"directed", "undirected", "max", "min", "upper", "lower", "plus"
    #'
    #' @return data.frame with graph metrics

    # Form Graph Adjacency matrix
    # ---------------------------
    system(paste("echo",session_name,"' current stage : start calculating FC graph","'"))
    FC.graph <- graph.adjacency(adj, weighted=weighted, mode=mode)
    FC.un_graph <- graph.adjacency(adj, weighted=TRUE, mode='undirected')    
    system(paste("echo",session_name,"' current stage : end calculating FC graph","'"))
    
    # Multistep calculations
    # ---------------------------
    # calculate Burt's constraint
    constrain = constraint(FC.graph, nodes = V(FC.graph), weights = NULL)
    constrain[is.nan(constrain)] = 0

    # graph diversity (scaled Shannon entropy of edges)
    diverse = diversity(FC.un_graph, weights = NULL, vids = V(FC.un_graph))
    diverse[is.nan(diverse)] = 0

    incident.byVertex = sapply(1:ncol(adj), incident, graph = FC.graph, mode = "total")
    neighbor.byVertex = sapply(1:ncol(adj), neighbors, graph = FC.graph, mode = "total")

    # average nearest neighbor degree of a vertex
    k.nn = knn(FC.graph, vids = V(FC.graph), weights = NULL)$knn
    k.nn[is.na(k.nn)] = 0

    ##################
    ## extract stuff 
    system(paste("echo",session_name,"' current stage : calculate graph matrices","'"))
    out = data.frame(
        session_id = session_name,
        # Global mean alpha centrality
        alpha.centrality = mean(alpha_centrality(FC.graph, nodes = V(FC.graph), alpha = 1, loops = FALSE,
                                                        exo = 1, weights = NULL, tol = 1e-07, sparse = TRUE)),
        # Global Assortativity
        assort = assortativity(FC.graph, types1 = c(1:ncol(adj)), types2 = NULL, directed = TRUE),
        authority = authority_score(FC.graph, scale = TRUE, weights = NULL, options = arpack_defaults)$value,
        automorph = automorphisms(FC.graph, sh = "fm")$group_size,
        between = mean(betweenness(FC.graph, v = V(FC.graph), directed = TRUE, weights = NULL,
                                        nobigint = TRUE, normalized = FALSE)),
        bi.connected = biconnected_components(FC.graph)$no,
        central.betw = centr_betw(FC.graph, directed = TRUE, nobigint = TRUE, normalized = TRUE)$centralization,
        central.betw_tmax = centr_betw_tmax(FC.graph, nodes = ncol(adj), directed = TRUE),
        central.close = centr_clo(FC.graph, mode = "total", normalized = TRUE)$centralization,
        central.close_tmax = centr_clo_tmax(FC.graph, nodes = 0, mode = "total"),
        central.degree = centr_degree(FC.graph, mode = "total", loops = TRUE, normalized = TRUE)$centralization,
        central.degree_tmax =centr_degree_tmax(FC.graph, nodes = 0, mode = "total", loops = FALSE),
        central.eigen = centr_eigen(FC.graph, directed = TRUE, scale = TRUE,
                                        options = arpack_defaults, normalized = TRUE)$centralization,
        central.eigen_tmax = centr_eigen_tmax(FC.graph, nodes = 0, directed = TRUE, scale = TRUE),
        num.cliques = clique_num(FC.graph),
        max.cliques = count_max_cliques(FC.graph),
        close = mean(closeness(FC.graph, vids = V(FC.graph), mode = "total",
                                    weights = NULL, normalized = FALSE)),
        num.component = components(FC.graph, mode = "strong")$no,
        constrain.mean = mean(constrain),
        core = mean(coreness(FC.graph, mode = c("all"))),
        count.motifs = count_motifs(FC.graph, size = 3, cut.prob = rep(0, 3)),
        count.triangles = mean(count_triangles(FC.graph, vids = V(FC.graph))),
        degree.mean = mean(degree(FC.graph, v = V(FC.graph), mode = "total", loops = TRUE, normalized = FALSE)),
        diameters = diameter(FC.graph, directed = TRUE, unconnected = TRUE, weights = NULL),
        mean.distance = mean_distance(FC.graph, directed = TRUE, unconnected = TRUE),
        all.shortest.paths = sum(all_shortest_paths(FC.graph, from = 1, to = ncol(adj), mode = "all", weights = NULL)$nrgeo),
        diverse.sum = sum(diverse),
        eccentric = mean(eccentricity(FC.graph, vids = V(FC.graph), mode = "total")),
        edge.connectivity = edge_connectivity(FC.graph, source = 1, target = ncol(adj), checks = TRUE),
        edge.density = edge_density(FC.graph, loops = TRUE),
        ego.size = mean(ego_size(FC.graph, 1, nodes = V(FC.graph), mode = "all", mindist = 0)),
        eigen.centrality = eigen_centrality(FC.graph, directed = TRUE, scale = TRUE, weights = NULL, options = arpack_defaults)$value,
        girth.value = girth(FC.graph, circle = TRUE)$girth,
        hub.score = hub_score(FC.graph, scale = TRUE, weights = NULL, options = arpack_defaults)$value,
        incidents = mean(incident.byVertex),
        k.nearestn = mean(k.nn),
        min.cut = min_cut(FC.graph, source = 1, target = ncol(adj), capacity = NULL, value.only = TRUE),
        motif = length(motifs(FC.graph, size = 3, cut.prob = rep(0, 3))),
            neighbor = mean(neighbor.byVertex),
        page.rank = mean(page_rank(FC.graph, algo = "prpack", vids = V(FC.graph), directed = TRUE, damping = 0.85, 
                                        personalized = NULL, weights = NULL, options = NULL)$vector),
        similarity.means = mean(colMeans(similarity(FC.graph, vids = V(FC.graph), 
                                                        mode = "total", loops = FALSE, method = "dice"))),

        strength.mean = mean(strength(FC.graph, vids = V(FC.graph), mode = "total", 
                                        loops = TRUE, weights = NULL)),

        transitivity.mean = transitivity(FC.graph, type = "global", vids = NULL, 
                                            weights = NULL, isolates = "zero")
    )
}

# command line arguments
option_list = list(
  make_option(c("-f", "--fc_file"), type="character", default=NULL, 
              help="Full path to functional connectivity csv", metavar="character"),
  make_option(c("-s", "--session_id"), type="character", default=NULL, 
              help="Session id, e.g. CU0011_baseline", metavar="character"),   
  make_option(c("-o", "--out_file"), type="character", default=NULL, 
              help="Full path to save graph metrics", metavar="character")  
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

#fc_file = opt$fc_file
#fc_file = '/fmri-qunex/research/imaging/datasets/embarc/qunex_features/EMBARC/refined/TX0039_baseline/functional/funccon/PearsonRtoZ/sub-TX0039_baseline_task-restrun_2_proc-Atlas_s_hpss_res-mVWMWB_lpss_frameCensor_study-EMBARC_atlas-CABNP_stat-PearsonRtoZ.csv.gz'
#session_id = 'CU0011_baseline'
#out_file = '/fmri-qunex/research/imaging/datasets/embarc/qunex_features/EMBARC/refined/TX0039_baseline/functional/funccon/graph_theory/test.csv.gz'

fc = read.csv(opt$fc_file, header=TRUE)
adj = form_adjacency(fc)
out = graph_metrics(adj, opt$session_id)

gz_out = gzfile(opt$out_file, 'w')
write.csv(out, gz_out, row.names=F)
close(gz_out)
