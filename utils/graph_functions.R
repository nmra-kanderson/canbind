library(igraph)
form_adjacency = function(fc, remove_diag = TRUE, quantile_threshold = 0.8) {
    adj <- as.matrix(fc)
    if(remove_diag) diag(adj) = 0
    adj[adj < quantile(adj, quantile_threshold) ] <- 0
    adj[adj > 0 ] <- 1
	return(adj)
}

graph_metrics = function(adj,session_name, weighted = TRUE, mode = "directed") {
  
  ## form graph
  system(paste("echo",session_name,"' current stage : start calculating FC graph","'"))
  FC.graph <- graph.adjacency(adj, weighted=weighted, mode=mode)
  system(paste("echo",session_name,"' current stage : end calculating FC graph","'"))
  #################
  ## multistep calculations
  constrain = constraint(FC.graph, nodes = V(FC.graph), weights = NULL)
  constrain[is.nan(constrain)] = 0
  
  diverse = diversity(FC.graph, weights = NULL, vids = V(FC.graph))
  diverse[is.nan(diverse)] = 0
  
  incident.byVertex = sapply(1:ncol(adj), incident, graph = FC.graph, mode = "total")
  neighbor.byVertex = sapply(1:ncol(adj), neighbors, graph = FC.graph, mode = "total")

  k.nn = knn(FC.graph, vids = V(FC.graph), weights = NULL)$knn
  k.nn[is.na(k.nn)] = 0
  
  ##################
  ## extract stuff 
  system(paste("echo",session_name,"' current stage : calculate graph matrices","'"))
  out = data.frame(
    alpha.centrality = mean(alpha_centrality(FC.graph, nodes = V(FC.graph), alpha = 1, loops = FALSE,
                                                 exo = 1, weights = NULL, tol = 1e-07, sparse = TRUE)),
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

##################
#### apply code
library(RNifti)
ss = function (x, pattern, slot = 1, ...) sapply(strsplit(x = x, split = pattern, ...), "[", slot)

project_path = "/home/ubuntu/fsx/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-one-scan/sessions/"
project_path = "/fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-one-scan/sessions/"

sample_manifest = read.delim("/fmri-qunex/research/imaging/datasets/embarc/processed_data/pf-pipelines/qunex-nbridge/studies/embarc-20201122-one-scan/info/bids/embarc-20201122-one-scan/participants.tsv")
scan_manifest = data.frame(SubjectID = rep(sample_manifest$SRC_SUBJECT_ID, each=6),
              Week = rep(c("baseline", "wk1"), each = 3, times = nrow(sample_manifest)),
              Session = rep(paste0("bold", 1:3), times = 2*nrow(sample_manifest)))
scan_manifest$nii = paste0(project_path, scan_manifest$SubjectID, "_", scan_manifest$Week, 
              "/images/functional/", scan_manifest$Session, 
              "_Atlas_s_hpss_res-mVWMWB_lpss_BOLD-CAB-NP-v1.0_r_Fz.csv")

## more info on session type
scan_manifest$Session_Type = "rest"
scan_manifest$Session_Type[scan_manifest$Session_Type == "bold1"] = "ert"

scan_manifest$nii

## make ID      
scan_manifest$ScanID = paste(scan_manifest$SubjectID, scan_manifest$Week, 
          scan_manifest$Session,sep="_")
          
## retain existing data 
table(file.exists(scan_manifest$nii)) 
scan_manifest= scan_manifest[file.exists(scan_manifest$nii),]

## files
niis = scan_manifest$nii 
names(niis) = scan_manifest$ScanID  

## iterate over samples
graph_list = pbmclapply(niis, function(f) {
  out <- tryCatch(
    {
        cat(".")
        fc = read.csv(f, header = FALSE)
        fc[,1] = as.numeric(ss(fc[,1], ": ", 2))
        adj = form_adjacency(fc)
        session_name <- strsplit(f, "/")
        bold_name <- strsplit(session_name[[1]][18], "_")
        out = graph_metrics(adj,paste("[",session_name[[1]][15],bold_name[[1]][1],"]"))
        return(out)},
        error=function(cond) {
            system(paste("echo",session_name,"' Error when generating graph metrics","'"))
            system(paste("echo",cond,"' Error when generating graph metrics","'"))
            return(NULL)
          })
      return(out)
}, mc.cores=12)


# Ignore failed calls resulting in NULL data frames
successful_calls = graph_list[lengths(graph_list) != 0]

# Combine outputs across samples
graph_measures = do.call("rbind", successful_calls)

# Save final results to a file
write.table(graph_measures, '~/fz_graph_metrics.csv')

## combine outputs across samples
graph_measures = do.call("rbind", graph_list)

