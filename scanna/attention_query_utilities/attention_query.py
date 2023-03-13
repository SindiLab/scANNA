"""Implementations of AttentionQuery class for interpreting attention values."""

from anndata import AnnData
from collections import Counter
import gseapy as gp
from itertools import chain
from ..model import ProjectionAttention, AdditiveModel
import numpy as np
import pandas as pd
import torch
from typing import List
from ..utilities import sparse_to_dense


class AttentionQuery():
    """ Class implementation for extracting and querying attention weights.

    This class contains the core methods for extracting cluster-specific
    attention weights, which are used for querying and interpretability.

    Attributes:
        data: The scanpy object that we want to make predictions on.
          Note: The dataframe can be changed or retrived via the defined
          "setter" and "getter" methods.
        split: The split of the scanpy data we want to use, e.g. "test" split.
        attention_weights: The attnetion weights extracted from the model.
        predicted: Model predictions over the chosen split.
        score: The gene score matrix.
        attentive_genes: Top genes with the highest weights (attention values).
        top_genes_df: A dataframe ranked based on the top attentive genes.

    """

    def __init__(self,
                 scanpy_object: AnnData,
                 model: ProjectionAttention | AdditiveModel = None,
                 split_test: bool = False,
                 which_split: str = "test"):
        """ Initializer of the AttentionQuery class.

        Args:

            scanpy_object: The scanpy object that contains the attention weights
              and predicted cells.
            model: The model we want to use to make predictions and extract
              attention weights from.
            split_test: A boolean indicating whether the user wants to get
              attention weights for the entire data (when "False"), or just a
              split (when set to "True").
            which_split: The name of the split that we are interested in and
              which exists in the AnnData.

        """
        if split_test:
            print("==> Splitting the data to 'test' only:")
            self.data = scanpy_object[scanpy_object.obs.split == which_split]
        else:
            self.data = scanpy_object

        self.split = split_test
        self._correct_predictions_only_flag = False
        self.model = model
        # Initializing the value for later use.
        self.ranked_genes_per_clust = None

    def get_gene_query(self,
                       model: ProjectionAttention |
                       AdditiveModel = None,
                       number_of_query_genes: int = 100,
                       local_scanpy_obj: AnnData = None,
                       attention_type: str = "additive",
                       inplace: bool = False,
                       mode: str = "tfidf",
                       correct_predictions_only: bool = True,
                       use_raw_x: bool = True,
                       verbose: bool = True):
        """ Class method with automated worflow of getting query genes.

        Args:
            model: The model we want to use to make predictions and extract
              attention weights from.
            number_of_query_genes: An integer indicating the number of top
              genes desired
            local_scanpy_obj: An AnnData object that would locally replace the
              scanpy object that was set in the constructor for the object.
            attention_type: The type of attention the inputted model was
              trained with. This will be 'additive' in most cases (even when
              scANNA included projection blocks).
            inplace: Wheather we want changes to be inplace, or on a copy (in
              which case it will be returned.
            correct_predictions_only: Whether to extract attention only from the
              correct predictions or not.
            use_raw_x: To use the "adata.raw.X" or just adata.X (depending on
              preprocessing pipeline).
            verbose: Whether the methods should print out their process or not.

        Returns:
            Based on the ranking method (i.e. TFIDF or averages), the method
            will return:

            (1) A dataframe with ranked gene names for each cluster
            (2) A dictionary mapping cluster names to the attention dataframe
                (the matrix of attention values for cells x genes)
            (3) A dictionary mapping each cluster to a series containing a gene
                list as index, mapped to the normalized attention values.

        Raises:
            None.
        """
        # As the first step, we want to extract attention values for each gene.
        _, _ = self.assign_attention(
            model=model,
            inplace=inplace,
            local_scanpy_obj=local_scanpy_obj,
            correct_predictions_only=correct_predictions_only,
            attention_type=attention_type,
            verbose=verbose,
            use_raw_x=use_raw_x)

        # Next, we calculate the top attentive genes in each cluster
        (clust_to_att_dict, top_genes_to_df_dict,
         top_n_names) = self.get_top_n_per_cluster(n=number_of_query_genes,
                                                   model=model,
                                                   mode=mode,
                                                   verbose=verbose)

        # Lastly, depending on the normalization method (e.g. TFIDF), we
        # normalize the attention values and return the top attentive genes for
        # querying.
        if mode.lower() == "tfidf" or mode.lower() == "tf-idf":
            tf_idf_df = self.calculate_tfidf(
                pd.DataFrame.from_dict(top_n_names),
                top_genes_to_df_dict,
                n_genes=number_of_query_genes)
            self.ranked_genes_per_clust_df = tf_idf_df

            if not inplace:
                return tf_idf_df, clust_to_att_dict, top_genes_to_df_dict
        else:
            self.ranked_genes_per_clust_df = pd.DataFrame.from_dict(top_n_names)
            if not inplace:
                return (pd.DataFrame.from_dict(top_n_names), clust_to_att_dict,
                        top_genes_to_df_dict)

    def make_enrichment_plots(
        self,
        number_of_genes_to_query: int = 50,
        which_cluster: int | str = None,
        clusters_to_names_dict: dict = None,
        number_of_total_clusters: int = None,
        species: str = "Human",
        return_results: bool = False,
        save: bool = False,
        where_to_save_plots: str = None,
        gene_sets: List[str] | str = "default",
        verbose: bool = True,
    ):
        """ Class method with automated worflow of getting query genes.

        Args:
            number_of_genes_to_query: An integer indicating how many genes we
              want to run enrichment test on (query)
            which_cluster: An optional integer or string indicating which
              cluster we want to annotate.
            clusters_to_names_dict: An optional dictionary containing the
              mapping between clusters and cell type names.
            number_of_total_clusters: An integer indicating the total number of
              clusters present in the data. This option should be provided when
              "which_cluster" is set to None.
            species: A string indicating the species that the sample is coming
              from (e.g. Human or Mouse).
            return_results: A boolean indicating whether we need the dataframe
              generated by the enrichment test.
            save: A boolean indicating whether we want to save results (e.g
              plots) externally or not.
            where_to_save_plots: An optional string consisting of the path for
              where the results are saved. The string passed must be a path and
              not including the name of the file (the ending string will be
              treated as a directory).
            gene_sets: Either a list of gene sets for the query (as strings), or
              the value "default" which will use default gene sets.
            verbose: A boolean for whether we want to print out a more complete
              dialogue.

        Returns:
            Based on user preference, either returns "None" or a dataframe
            containing the enrichmeht test results.

        Raises:
            ValueError: If neither of the arguments "which_cluster" nor
              "number_of_total_clusters" are provided when calling the method.
        """
        if gene_sets == "default":
            # We use Azimuth as the base database for humans
            if species.lower() == "human":
                genesets = ["Azimuth_Cell_Types_2021"]
            elif species.lower() == "mouse" or species.lower() == "mice":
                # Making sure the species name is mouse so that it works with
                # the EnrichR API
                species = "mouse"
                # We use Tabula Muris as the main database for mice
                genesets = ["Tabula_Muris"]
            # And, we add the following two data basis for additional
            # verification
            else:
                print("==> Did not detect 'mice' or 'humans' as species."
                      " We highly encourage specifying the gene sets explicitly"
                      " for other species (or specific use cases).")
                genesets = []

            genesets.extend(
                ["CellMarker_Augmented_2021", "PanglaoDB_Augmented_2021"])

        if which_cluster is not None:
            which_cluster = int(which_cluster)
            cluster = "Cluster_" + f"{which_cluster}"
            if clusters_to_names_dict is None:
                clusters_to_names_dict = (f"{cluster}:Unidentified Celltype"
                                          f"{which_cluster}")
            try:
                gene_list = self.ranked_genes_per_clust_df[
                    f"{cluster}_genes"].tolist()[:number_of_genes_to_query]
            except KeyError as _:
                gene_list = self.ranked_genes_per_clust_df[f"{cluster}"].tolist(
                )[:number_of_genes_to_query]
            if verbose:
                print("==> Performing enrichment test on top "
                      f"{number_of_genes_to_query} genes.")
            if save:
                if where_to_save_plots is None:
                    where_to_save_plots = "./Enrichment_Plots/"
                else:
                    where_to_save_geneset_plots = (
                        f"{where_to_save_plots}"
                        f"{clusters_to_names_dict[cluster]}"
                        "_geneset_score.png")

                    where_to_save_combine_score_plots = (
                        f"{where_to_save_plots}"
                        f"{clusters_to_names_dict[cluster]}"
                        "_combined_score.png")
            else:
                # Making sure even if the user has provided a path, EnrichR does
                # not save any results.
                where_to_save_plots = None
                where_to_save_geneset_plots = None
                where_to_save_combine_score_plots = None
            # Perforning the erichment test using EnrichR
            enrich_test = gp.enrichr(
                gene_list=gene_list,
                gene_sets=genesets,
                organism=species,
                outdir=where_to_save_plots,
            )
            # cleaning up the string results before plotting
            enrich_test = self._clean_enrichment_results(enrich_test)
            print(type(enrich_test))
            if verbose:
                print("    -> Results per gene set:")
            _ = gp.dotplot(
                enrich_test.results,
                column="Adjusted P-value",
                x="Gene_set",
                size=6,
                top_term=5,
                figsize=(4, 6),
                title=f"Enrichment for {clusters_to_names_dict[cluster]}",
                ofname=where_to_save_geneset_plots,
                xticklabels_rot=45,
                show_ring=True,
                marker="o",
            )

            if verbose:
                print("    -> Results for combine score:")

            _ = gp.dotplot(
                enrich_test.results,
                column="Adjusted P-value",
                x="Combine Score",
                size=6,
                top_term=5,
                figsize=(4, 6),
                title=f"Enrichment for {clusters_to_names_dict[cluster]}",
                ofname=where_to_save_combine_score_plots,
                xticklabels_rot=45,
                show_ring=True,
                marker="o",
            )
            if return_results:
                return enrich_test.results

        else:
            if number_of_total_clusters is None:
                raise ValueError(
                    "Please provide either the cluster of interest"
                    " ('which_cluster' arg) or the total number of"
                    " clusters ('number_of_total_clusters' arg) for"
                    " the enrichment analysis.")
            for cluster in range(number_of_total_clusters):
                print(f"==> Analysis for cluster {cluster}:")
                self.make_enrichment_plots(
                    number_of_genes_to_query=number_of_genes_to_query,
                    which_cluster=cluster,
                    return_results=False,
                    clusters_to_names_dict=clusters_to_names_dict,
                    number_of_total_clusters=number_of_total_clusters,
                    species=species,
                    save=save,
                    where_to_save_plots=where_to_save_plots,
                    verbose=verbose)
            print(">-< Done.")

    def get_important_global_genes(self,
                                   model: ProjectionAttention |
                                   AdditiveModel = None,
                                   how_many_global_genes: int = 50,
                                   split_data: bool = True,
                                   local_scanpy_obj: AnnData = None,
                                   attention_type: str = "additive",
                                   inplace: bool = False,
                                   rank_mode: str = "mean",
                                   correct_predictions_only: bool = True,
                                   use_raw_x: bool = True,
                                   verbose: bool = True):
        """ Class method with automated worflow of getting query genes.

        Args:
            model: The model we want to use to make predictions and extract
              attention weights from.
            how_many_global_genes : An integer indicating the number of top
              genes desired
            split_data: A boolean indicating whether the annotated data should
              be splitted into train and test.
            local_scanpy_obj: An AnnData object that would locally replace the
              scanpy object that was set in the constructor for the object.
            attention_type: The type of attention the inputted model was
              trained with. This will be 'additive' in most cases (even when
              scANNA included projection blocks).
            inplace: Wheather we want changes to be inplace, or on a copy (in
              which case it will be returned.
            correct_predictions_only: Whether to extract attention only from the
              correct predictions or not.
            use_raw_x: To use the "adata.raw.X" or just adata.X (depending on
              preprocessing pipeline).
            verbose: Whether the methods should print out their process or not.

        Returns:
            Based on the data splitting ("data_split" arg set to true or false),
            the method will return two different set of outputs:

            (1) If "data_split" is True, then the method will output six
                analysis outputs based on the most important genes identified.
                These outputs consist of values, labels and cell type names for
                train and test split (equalling to six).
            (2) If "data_split" is True then the returned outputs are the
                values, labels and cell type names for all cells (thus three
                outputs).

        Raises:
            None.
        """
        # As the first step, we want to extract attention values for each gene.
        att_adata, _ = self.assign_attention(
            model=model,
            inplace=inplace,
            local_scanpy_obj=local_scanpy_obj,
            correct_predictions_only=correct_predictions_only,
            attention_type=attention_type,
            verbose=verbose,
            use_raw_x=use_raw_x)

        # Next, we extract the top attentive gene names across all cells
        top_global_genes = self.get_top_n(n=how_many_global_genes,
                                          rank_mode=rank_mode,
                                          verbose=verbose).index.tolist()
        # Now subsetting the dataset to only include the top genes
        top_genes_subset_adata = att_adata[:, top_global_genes]

        if split_data:
            top_genes_subset_train = top_genes_subset_adata[
                top_genes_subset_adata.obs.split == "train"]
            top_genes_subset_test = top_genes_subset_adata[
                top_genes_subset_adata.obs.split == "test"]
            # Separating diffrent components for further analysis and validation
            subset_train = sparse_to_dense(top_genes_subset_train)
            subset_labels_train = top_genes_subset_train.obs.cluster.to_numpy()
            subset_names_train = list(
                top_genes_subset_train.obs.celltypes.to_numpy())

            subset_test = sparse_to_dense(top_genes_subset_test)
            subset_labels_test = top_genes_subset_test.obs.cluster.to_numpy()
            subset_names_test = list(
                top_genes_subset_test.obs.celltypes.to_numpy())

            return (subset_train, subset_test, subset_labels_train,
                    subset_labels_test, subset_names_train, subset_names_test)

        else:
            # Separating diffrent components for further analysis and validation
            subset_all = sparse_to_dense(top_genes_subset_adata)
            subset_labels_all = top_genes_subset_adata.obs.cluster.to_numpy()
            subset_names_all = list(
                top_genes_subset_adata.obs.celltypes.to_numpy())

            return subset_all, subset_labels_all, subset_names_all

    def assign_attention(self,
                         model: ProjectionAttention |
                         AdditiveModel = None,
                         local_scanpy_obj: AnnData = None,
                         attention_type: str = "additive",
                         inplace: bool = False,
                         correct_predictions_only: bool = True,
                         use_raw_x: bool = True,
                         device: str = "infer",
                         verbose: bool = True):
        """ The method to assign attention score to a scanpy object.

        Args:
            model: The model we want to use to make predictions and extract
              attention weights from.
            local_scanpy_obj: An AnnData object that would locally replace the
              scanpy object that was set in the constructor for the object.
            attention_type: The type of attention the inputted model was
              trained with. This will be 'additive' in most cases (even when
              scANNA included projection blocks).
            inplace: Wheather we want changes to be inplace, or on a copy (in
              which case it will be returned.
            correct_predictions_only: Whether to extract attention only from the
              correct predictions or not.
            use_raw_x: To use the "adata.raw.X" or just adata.X (depending on
              preprocessing pipeline).
            verbose: Whether the methods should print out their process or not.

        Returns:
            The method will return:

            (1) A scanpy object with predictions and attention weights as
                added keys to the AnnData object.

            (2) A dataframe consisting of genes as columns and attention weights
                as the corresponding row values.

        Raises:
            ValueError: An error occured during reading the trained model.

        """
        if model is None and self.model is None:
            raise ValueError("Please provide a model for making predictions and"
                             " extracting attention weights.")
        elif model is None and self.model is not None:
            model = self.model

        if device == "infer":
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        # set the flag
        self.attention_weights = True

        if local_scanpy_obj is None:
            test_data = self.data.copy()
            if verbose:
                print("*Caution*: The method is running on the entire data."
                      " If this is not what you want, provide scanpy"
                      "object.")
        else:
            test_data = local_scanpy_obj

        if use_raw_x:
            test_tensor = torch.from_numpy(sparse_to_dense(test_data.raw))
        else:
            test_tensor = torch.from_numpy(sparse_to_dense(test_data))

        if verbose:
            print("==> Calling forward:")

        model.eval()

        if verbose:
            print("    -> Making predictions")

        with torch.no_grad():
            model.to(device)
            logits, score, attentive_genes = model(
                test_tensor.float().to(device), training=False)
            _, predicted = torch.max(logits.squeeze(), 1)

        # If this call is for the entirety of the data we have, then we should
        # assign all cells, otherwise we would get a dimension error if we are
        # considering a subset.
        predicted = predicted.detach().cpu().numpy()
        score = score.detach().cpu().numpy()
        if attention_type == "multi-headed":
            score = score.reshape(int(score.shape[0] / 8), 8)
        attentive_genes = attentive_genes.detach().cpu().numpy()

        if local_scanpy_obj is None:
            self.predicted = predicted
            self.score = score
            self.attentive_genes = attentive_genes
            if verbose:
                print("    -> Assigning attention weights globally: ")
            test_data.obsm["attention"] = score  #attentive_genes
            predicted_str = [f"{i}" for i in predicted]
            test_data.obs["prediction"] = predicted_str
            test_data.obs["prediction"] = test_data.obs["prediction"].astype(
                "str"
            )  # changed to str from category since it was causing issues
            # adding a check for the correct data type in the cluster column
            if test_data.obs["cluster"].dtype != str:
                test_data.obs["cluster"] = test_data.obs["cluster"].astype(
                    "str")

        if correct_predictions_only:
            if verbose:
                print("    -> **Returning only the correct predictions**")
            test_data = test_data[test_data.obs["cluster"] ==
                                  test_data.obs["prediction"]]
            # We will use this adata later on instead if we only want to look
            # at the correct predictions.
            self.correct_pred_adata = test_data
            self._correct_predictions_only_flag = True

        if verbose:
            print("    -> Creating a [Cells x Attention Per Gene] DataFrame")

        try:
            att_df = pd.DataFrame(test_data.obsm["attention"].values,
                                  index=test_data.obs.index,
                                  columns=test_data.var.gene_ids.index)
        except AttributeError as _:
            att_df = pd.DataFrame(test_data.obsm["attention"],
                                  index=test_data.obs.index,
                                  columns=test_data.var.index)

        if local_scanpy_obj is None:
            self.att_df = att_df

        if inplace:
            if verbose:
                print(
                    "    -> Making all changes inplace and returning input data"
                    " with changes")
            self.data = test_data
            return self.self.data, att_df

        else:
            if verbose:
                print("    -> Returning the annData with the attention weights")
            return test_data, att_df

    def get_top_n(self,
                  n: int = 100,
                  dataframe: pd.DataFrame = None,
                  rank_mode: str = None,
                  verbose: bool = True):
        """ Class method for getting the top n genes for the entire dataset.

        Args:
            n: An integer indicating the number of top genes desired.
            dataframe: The dataframe we want to find the top n Genes in.
            rank_mode: The mode we want to use for ranking top "n" genes.
            verbose: If we want to print out a complete dialogue.

        Returns:
            A dataframe containing the top n genes with the shape cells x genes.

        Raises:
           NotImplementedError: An error occured if "rank_mode" argument is not
             one of the existing modes.

        """

        if not hasattr(self, "attention_weights"):
            print("Please first set the attention weights by calling"
                  "AssignAttention()")
            return 0

        if verbose:
            print(f"==> Getting Top {n} genes for all cells in the original"
                  " data")
            print("    -> Be cautious as this may not be cluster specific."
                  " If you want cluster specific, pleae call"
                  " 'get_top_n_per_cluster()' method.'")

        # make it to be genes x cells
        if dataframe is None:
            att_df_trans = self.att_df.T

        else:
            att_df_trans = dataframe.T

        # Finding n largest genes (features based on the mode:
        if rank_mode is not None:
            if verbose:
                print(f"    -> Ranking mode: {rank_mode}")
            if rank_mode.lower() == "mean":
                top_genes_transpose = att_df_trans.loc[att_df_trans.sum(
                    axis=1).nlargest(n, keep="all").index]

            elif rank_mode.lower() == "nlargest":
                top_genes_transpose = att_df_trans.nlargest(
                    n, columns=att_df_trans.columns, keep="all")

            else:
                raise NotImplementedError(f"Your provided mode={rank_mode} has"
                                          "not been implemented yet. Please"
                                          " choose between 'mean' or 'None' for"
                                          "now.")

        else:
            top_genes_transpose = att_df_trans.nlargest(
                n, columns=att_df_trans.columns, keep="all")

        # return the correct order, which is cells x genes
        if dataframe is None:
            self.top_genes_df = top_genes_transpose
            return self.top_genes_df

        else:
            return top_genes_transpose.T

    def get_top_n_per_cluster(self,
                              n: int = 25,
                              model=None,
                              mode: str = "tfidf",
                              top_n_rank_method: str = "mean",
                              verbose: bool = False):
        """ Get the top n genes for each indivual cluster.

        Args:
            n: An integer indicating the number of top genes to keep.
            model: The model we want to use to make predictions
            mode: The mode we want to use for identifying top genes and
              normalizing the values
            top_n_rank_method: The mode we want to use for ranking top n genes
              (the mode used for get_top_n method different than "mode"
              argument) for this method.

        Returns:

        This method's reutns are "mode" dependent, and will return three
        objects:

        (1) A dictionary mapping each cluster to the attention scores.

        (2) A dictionary containing the sum of gene attention scores for each
            cluster.

        (3) A dictionary mapping containing the name of top n genes for each
            cluster.

        Raises:
            ValueError: An error occured during reading the trained model.

        """
        if model is None and self.model is None:
            raise ValueError("Please provide a model for making predictions and"
                             " extracting attention weights.")

        print(f"==> Top {n} genes will be selected in {mode} mode")

        # Dictionary to map clusters to their attention weights, no ranking
        # or filtering.
        self.clust_to_att_dict = {}
        # Dictionary to map clusters to series containing their summed attention
        # weights per gene.
        self.clust_sums_dict = {}
        # Dictionary to mapping clusters to dataframes containing the top n
        # genes and their attention weights.
        self.top_n_df_dict = {}
        # dictionary to mapping clusters to a dataframe containing the top n
        # genes based on their summed attention weights.
        self.top_n_names_dict = {}

        if self._correct_predictions_only_flag:
            data_to_use = self.correct_pred_adata
        else:
            data_to_use = self.data

        print(f"==> Getting Top {n} genes for each cluster in the data")
        iter_list = list(data_to_use.obs.cluster.unique())
        iter_list.sort()

        for i in iter_list:
            if verbose:
                print(f"    -> Cluster {i}:")
            # get data for the current cluster
            curr_clust = data_to_use[data_to_use.obs.cluster == i]
            if verbose:
                print(f"    -> Cells in current cluster: {curr_clust.shape[0]}")

            # Getting the cell x attention per gene df for the current cluster.
            curr_att_df = self.att_df.loc[curr_clust.obs.index]
            # Mapping the clusters to the attention dataframe.
            self.clust_to_att_dict[f"Cluster_{i}"] = curr_att_df
            # Getting the top n gene dataframe based on the mode.
            if mode.lower() == "tfidf":
                self.clust_sums_dict[f"Cluster_{i}"] = curr_att_df.T.sum(axis=1)
            else:
                self.top_n_df_dict[f"Cluster_{i}"] = self.get_top_n(
                    n=n,
                    dataframe=curr_att_df,
                    verbose=False,
                    rank_mode=top_n_rank_method)

            # Gettting the top n gene names in ranked order (frp, highest
            # expression to lowest).
            self.top_n_names_dict[f"Cluster_{i}"] = curr_att_df.T.sum(
                axis=1).nlargest(n).index
            if verbose:
                print(f"    >-< Done with Cluster {i}")
                print()
        print(">-< Done with all clusters")

        if mode.lower() == "tfidf":
            return (self.clust_to_att_dict, self.clust_sums_dict,
                    self.top_n_names_dict)

        else:
            # TODO: Check the ranking and ordering on the returns
            return (self.clust_to_att_dict, self.top_n_df_dict,
                    self.top_n_names_dict)

    def make_values_unique(self,
                           top_n_dictionary: dict = None,
                           threshold: int = None,
                           verbose: bool = False):
        """ Class method to make all the values in a dictionary unique.

        Args:
            top_n_dictionary: The dictionary containing the top n gene names for
              all populations (or smaller populations)
            threshold: The threshold for removing common genes: if a gene occurs
              in threshold many populations, it will be removed.

        Returns:
            The modified *unique* top_n_dictionary dictionary based on the
            threshold provided.

        Raises:
            None.
        """

        if top_n_dictionary is None:
            print("==> Since no dictionary was provided, we will use gene names"
                  " dictionary as default")
            att_dict = self.top_n_names_dict.copy()
        else:
            att_dict = top_n_dictionary

        # concat all the gene names into a list
        all_genes = list(chain.from_iterable(att_dict.values()))

        if threshold is None:
            # do not threshold the allowed overlaps
            if verbose:
                print("    -> No thresholding... setting overlap bound to inf")
            threshold = np.inf

        # find duplicates that appear as many times as the threshold
        duplicate_list = [
            item for item, count in Counter(all_genes).items()
            if count > threshold
        ]
        print(f"==> Found {len(duplicate_list)} many duplicates that appear in"
              " more than {threshold} cluster(s)")

        for key in att_dict.keys():
            att_dict[key] = [
                item for item in att_dict[key] if item not in duplicate_list
            ]

        if top_n_dictionary is None:
            self.global_unique_gene_names = att_dict

        return att_dict

    def calculate_tfidf(self,
                        top_n_names: dict,
                        top_n_df_dict: dict,
                        n_genes: int = 100,
                        verbose: bool = False):
        """ Implementation of term frequency–inverse document frequency.

        This function aims to provide a list of top genes that have been
        normalized by term-frequency (TF) - inverse document frequency (IDF).
        We define the corpus-wide tf-idf score for each gene as
                            (1 - Sum(tf-idf)/log(N)),
        where N is the total number of pseudo documents (list of top genes for
        each cluster). The returned list of top genes are then sorted in
        descending order according to the tf-idf scores and subsequently
        returned.

        Args:
            top_n_names: A dictionary mapping containing the name of top n genes
              for each cluster.
            top_n_df_dict: A dictionary containing the sum of gene attention
              scores for each cluster.
            n_genes: A number indicating the length for return values.

        Returns:
            A pandas dataframe with the Corpus-Wide tf-idf scores for each gene,
            indexed by the genes, sorted in descending order by the tf-idf
            scores.

        Raises:
            None.
        """

        # Here we convert the intial inputs to dataframes.
        # Below is the dataframe containing the top n genes for each cluster.
        top_genes_df = pd.DataFrame.from_dict(top_n_names)
        # Below is the dataframe containing the sum of gene attention scores
        # for each cluster
        attention_sums = pd.DataFrame.from_dict(top_n_df_dict)
        series = pd.Series(top_genes_df.values.tolist())
        number_documents = series.shape[0]
        gene_counts = series.apply(lambda x: Counter(x))

        # Creating a new series to store the tf values.
        term_freq = gene_counts.apply(
            lambda x:
            {gene: count / sum(x.values()) for gene, count in x.items()})
        # Create a new series to store the idf values.
        idf = series.apply(
            lambda x: {
                gene: np.log(number_documents / len(
                    [i for i in series if gene in i])) for gene in set(x)
            })
        # Create a new series to store the tf-idf values
        tf_idf = term_freq.apply(
            lambda x: {
                gene: count * idf[term_freq[term_freq == x].index[0]][gene]
                for gene, count in x.items()
            })

        # Create a dataframe containing the tf-idf values.
        tfidf_normalized_series = 1 - pd.DataFrame(
            tf_idf.tolist()).sum() / np.log(number_documents)

        # Get tf-idf genes to index the attention sum dataframe
        tfidf_genes = tfidf_normalized_series.index.tolist()

        # Subset the dataframe to only contain the top genes
        most_attentive_subset = attention_sums.loc[tfidf_genes, :]

        # Perform element-wise muliplication
        attentive_tfidf_normalized = most_attentive_subset.mul(
            tfidf_normalized_series, axis=0)

        # Dictionary for storing each cluster tf-idf normalized values
        tfidf_norm = {}
        iter_list = list(attentive_tfidf_normalized.columns)

        # Loop through the clusters and pull genes and values
        for i in iter_list:
            if verbose:
                print(f"    -> {i}:")
            # get data for the current cluster
            cluster_df = attention_sums.loc[:, i]

            # get the top n gene names in ranked order (highest to lowest)
            tfidf_norm[f"{i}_genes"] = cluster_df.nlargest(n_genes).index
            tfidf_norm[f"{i}_tfidf_values"] = cluster_df.nlargest(
                n_genes).values

        # Create a dataframe from the dictionary
        tfidf_out = pd.DataFrame.from_dict(tfidf_norm)
        return tfidf_out

    # Getter and setter methods for the class.
    def get_scanpy_object(self):
        """Getter method for returning Scanpy object at any given time.

        Args:
            None.

        Returns:
            The current scanpy data set as the internal attribute.

        Raises:
            None
        """
        return self.data

    def set_scanpy_object(self, new_data):
        """Setter method for setting Scanpy object at any given time

        Args:
            new_data: A new scanpy object to replace the previously passed data.

        Return:
            None. Method will set the new dataset using the passed scanpy
            object.

        Raises:
            None.

        """
        self.data = new_data


# Internal utility method.

    def _clean_enrichment_results(self, enrichment_test: gp.Enrichr):
        """Internal utility method for cleaning up results from certain sets.

        Args:
            enrichment_test: The enrichment test object for the analysis
              performed.

        Returns:
            The same object as passed on without the additional strings in the
            "Terms" attribute of the object.

        Raises:
            None
        """
        enrichment_test.results.Term = enrichment_test.results.Term.str.split(
            " CL").str[0]
        return enrichment_test
