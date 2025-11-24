"""
---------------- UniversalPipeline objects - does some data validation and holds the encoder processor objs ----------
-------------------- purpose is to track variables and indices between raw and encoded data -----------------------
"""

import concurrent.futures
import gc
import logging
import multiprocessing
import os
import pickle
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import scipy
# import sentencepiece as spm
from numpy.typing import NDArray

from encoders import Processor

log = logging.getLogger(__name__)


def convert_datastructure_for_processing(data_object) -> pd.DataFrame:
    """
    Conforms incoming data into the DF format we're expecting
    Parameters
    ----------
    data_object : dict, array, series of df of data

    Returns
    -------
    DataFrame structured from the input data
    """
    if isinstance(data_object, pd.DataFrame):
        # early terminate? if it's a df, values are already matched.
        return data_object
    elif isinstance(data_object, dict):
        variables = [str(k) for k in data_object.keys()]
        all_values = [vals for vals in data_object.values()]
    elif isinstance(data_object, pd.Series):
        variables = [str(data_object.name)]
        all_values = data_object.values.reshape(1, -1)
    elif isinstance(data_object, np.array):
        variables = [str(i) for i in range(0, data_object.shape[0])]
        all_values = data_object

    variable_count = len(variables)
    all_lengths = [len(v) for v in all_values]

    # assertions necessary:
    assert variable_count == len(all_lengths)
    assert np.all(np.equal(all_lengths, all_lengths[0]))

    return pd.DataFrame.from_dict({k: v for k, v in zip(variables, all_values)})


def expand_list_col(results: dict, meta_dict: dict) -> pd.DataFrame:
    """
    Take the df with encoded objects and flatten out any multi-dimensional columns.
    eg for a text column, we'll have an encoded list of ints in one dataframe cell
    Parameters
    ----------
    results: the result data from the encoding process
    meta_dict: the metadata dictionary from the pipeline

    Returns
    -------
    dataframe with multidimensional columns expanded (flattened)
    """
    # print(f"unexpanded columns have {df.shape}")
    list_of_dicts = []
    for metadata in meta_dict.values():
        out = metadata["output_dimension"]
        if out == len(metadata["variable_names"]):
            for v_name in metadata["variable_names"]:
                if v_name in results.keys():
                    list_of_dicts.append({v_name: results[v_name]})
                    continue
                else:
                    raise KeyError(f"{v_name} not present in our result data")  # noqa
        else:  # we have a mismatch (meaning we have a one to many, like text encoder
            (v_name,) = metadata["variable_names"]
            num_columns = metadata["output_dimension"]
            _results = np.array(results.pop(v_name))
            for i in range(0, num_columns):
                list_of_dicts.append({f"{v_name}_{i}": _results[:, i]})

    df = pd.DataFrame({k: v for one_d in list_of_dicts for k, v in one_d.items()})

    return df


def combine_futures(futures) -> dict:
    """
    helper function for multithreading / multiprocessing the encoding process
    Parameters
    ----------
    futures : future processes created with multiprocessing

    Returns
    -------
    dict of the results
    """
    combined = {}
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        if isinstance(result, bool):
            continue
        combined.update(result)  # Combine the result into a single dict
    return combined


class UniversalPipeline:
    def __init__(self):
        """
        The Universal Pipeline is the container class for individual encoders - it holds the forward encoding pass
        methods, imputation, and serialization / deserialization mechanisms
        Instantite a pipeline with this class, then use add_encoder() to add column by column. Then finalize the
        pipeline and fit the encoders.

        """
        self.encoders = OrderedDict()
        self._order = {}
        self.frozen = False

        self._encoders: list[tuple[int, str, Processor]] = []

        self.nan_idxs = []


    def add_encoder(self, new_encoder: Processor) -> None:
        """
        add an unfitted encoder to the pipeline object - be sure to use the column's target and index
        eg
        pipeline.add_encoder(categorical_encoders.CategoricalProcessor(
            target="prof_work_role", col_idx=23,
            rare_encoding_threshold=0.01,
            min_classes_for_rare_qualification=9))
        Parameters
        ----------
        new_encoder : the instantiated Encoder

        Returns
        -------
        None, an inplace operation
        """
        if self.frozen:
            raise ValueError("pipeline is already frozen...")
        name = str(new_encoder.target)
        idx = int(new_encoder.variable_idx)
        self._encoders.append((idx, name, new_encoder))
        print(f"added {new_encoder}")

    def finalize_setup(self) -> bool:
        """
        Freeze the current added encoders - this is the step before fitting.
        (inplace operation)
        Returns
        -------
        bool - True if successful
        """
        for e_i, (idx, name, proc) in enumerate(sorted(self._encoders, key=lambda x: x[0])):
            self._order[idx] = name
            if e_i != idx:
                raise ValueError(
                    f"variables incides don't align -- did you miss one? {e_i} != {idx}"
                )
            self.encoders[name] = proc
        self.frozen = True
        del self._encoders
        return True

    def validate_df(self, df: pd.DataFrame) -> bool:
        """
        Compares the dataframe and the encoders we have added to the pipeline - matching the index and column name
        with the encoder's target column name and target index.

        if there are mismatches, there will be assertion errors when validating.

        Parameters
        ----------
        df : Dataframe of input pre-encoding data

        Returns
        -------
        boolean - True if dataframe and encoders align
        """
        cols = [(idx, name) for idx, name in enumerate(df.columns)]
        for idx, name in cols:
            try:
                _p = self.encoders[str(name)]
            except KeyError:
                _p = list(self.encoders.values())[int(idx)]
            assert (_p.variable_idx == idx) and (_p.target == name)
        return True

    @property
    def metadata(self) -> dict:
        """
        gets the metadata for each encoder
        Returns
        -------
        returns the {target: metadata_dict}, which has some variation per encoder type.
        """
        metalist = [v.metadata for k, v in self.encoders.items()]
        return {k: v for one_meta in metalist for k, v in one_meta.items()}

    @property
    def output_mapping(self) -> dict:
        """
        Parses the encoder metadata and returns the number of output dimensions - for some encoder types (like text
        or chronological encoders), we might have one input variable to many output variables. This will help us
        track the input columns / indices :: output indices
        Returns
        -------
        the target variable name (colname : (start_idx, end_idx) of the final array output
        """
        output_mapping = {}
        metalist = [v.metadata for v in self.encoders.values()]
        for values in metalist:
            for target, meta_dict in values.items():
                output_mapping.update({target: (meta_dict["idx"], meta_dict["output_dimension"])})

        return output_mapping

    def decode(self, input_data: dict) -> dict:
        """
        decode model predictions (class numbers) back into our target variables
        Parameters
        ----------
        input_data : prediction data from the model to decode back to the original representation in the training data.


        Returns
        -------
        dict or dataframe of the decoded predictions with any decoded targets included
        """
        tasks = {}
        for enc_i, (vari_name, encoder) in enumerate(self.encoders.items()):
            to_encode = input_data.get(vari_name, []).astype(np.object_)
            _reverse_encoding = encoder.inverse(values=to_encode)
            tasks.update({vari_name: _reverse_encoding})
        # return pd.DataFrame.from_dict(tasks)
        return tasks

    def encode(
        self, input_data: iter, impute: bool = False, use_multiprocess: bool = False
    ) -> NDArray:
        """
        If you run into issues, first try setting multiprocess to False to get more visibility. Multiprocessing
        always makes it more challanging to track down the issue!

        1) convert input data into a pandas DF if it's an array or dict of iterables
        2) validate the structures of the dataframe and encoders - make sure we can process these without additional
        remapping!
        3) track any NaN values for later imputation
        4) Proess the df through the fitted encoders



        Parameters
        ----------
        input_data : array or dataframe to encode
        use_multiprocess : bool - True to use multiprocessing for parallel encoding - useful when you have a very
        large number of columns (100+)

        Returns
        -------
        Dataframe of encoded values
        """

        if self.is_fitted is False:
            raise ValueError("Not yet fitted - unable to process")
        # convert structure into Dataframe, and validate
        _data = convert_datastructure_for_processing(input_data)
        success = self.validate_df(_data)
        if success is False:
            raise ValueError("datastructure did not match the expected data")

        self.df_nan_idxs = pd.isnull(_data).values
        tasks = {}
        if use_multiprocess:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=multiprocessing.cpu_count() // 2,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                for vari_name, encoder in self.encoders.items():
                    if encoder.additional_targets is None:
                        tasks.update({executor.submit(encoder.encode, _data[vari_name]): vari_name})
                    else:
                        dict_targs = encoder.additional_targets
                        additional_kwargs = {k: _data[v] for k, v in dict_targs.items()}
                        tasks.update(
                            {
                                executor.submit(
                                    encoder.encode, _data[vari_name], **additional_kwargs
                                ): vari_name
                            }
                        )

                completed = combine_futures(tasks)

        else:
            for vari_name, encoder in self.encoders.items():
                # print(f"encoding {vari_name}")
                if encoder.additional_targets is None:
                    tasks.update(encoder.encode(_data[vari_name]))
                else:
                    dict_targs = encoder.additional_targets
                    additional_kwargs = {k: _data[v] for k, v in dict_targs.items()}
                    tasks.update(encoder.encode(_data[vari_name], **additional_kwargs))
            completed = tasks

        encoded = expand_list_col(completed, meta_dict=self.metadata)
        log.debug("Completed Encoding")


        return encoded

    def fit(self, input_data: iter) -> bool:
        """
        Make sure that the data is in expected formats and alignment;
        Parameters
        ----------
        input_data : our input data to fit our encoders to - we'll validate and check alignment before we try fitting the encoder objects

        Returns
        -------
        Bool if successfully fitted
        """
        if not self.frozen:
            print("make sure you have run finalize_setup()")
            raise ValueError("Pipeline is not frozen, use finalize_setup")

        # validate columns_idx
        _data = convert_datastructure_for_processing(input_data)
        success = self.validate_df(_data)

        if success:
            fitted = []
            for target_name, encoder in self.encoders.items():
                if encoder.additional_targets is None:
                    fitted.append(encoder.fit(_data[target_name]))
                else:
                    dict_targs = encoder.additional_targets
                    additional_kwargs = {k: _data[v] for k, v in dict_targs.items()}
                    fitted.append(encoder.fit(_data[target_name], **additional_kwargs))

        if np.all(np.equal(fitted, True)):
            print("successfully fitted all encoders")
        else:
            raise ValueError("FITTING ISSUE")

        return True

    def _sort_encoders(self) -> None:
        """
        Inplace operation - On load, we need to make sure that the encoders are ordered correctly. So call this.
        """
        self.encoders = OrderedDict(
            {k: v for k, v in sorted(self.encoders.items(), key=lambda item: item[1].variable_idx)}
        )

    @property
    def output_column_names(self) -> list[str]:
        """
        Get the names of the flattened encoded array - necessary as the input data can be represented in the encoded
        data as one column : many columsn
        Returns
        -------
        list of strings
        """
        # assumptions: that our encoders are in their expected / sorted order
        tracker: int = 0
        current_index: int = 0
        output_columns: list = []

        for meta_dict in self.metadata.values():
            assert tracker >= meta_dict["idx"]
            tracker: int = meta_dict["idx"] + 1
            outs_count: int = meta_dict["output_dimension"]
            # variable_idx_range: tuple = (current_index, current_index + outs_count)
            variable_names: list = meta_dict["variable_names"]
            if len(variable_names) != outs_count:
                # we'll need to extend the name to include additional columns (should only apply to text encoders now)
                variable_names = [f"{variable_names[0]}_{idx}" for idx in range(0, outs_count)]

            output_columns.extend(variable_names)
            current_index += outs_count

        return output_columns

    @property
    def get_profiler_output_info(self) -> list[int]:
        """
        get the indices for the profiler question variables in the encoded data, and return a list of those indicies
        Returns
        -------
        list of indices in the encoded data array that are profiler questions
        """
        categorical_details = list[int]
        current_index: int = 0
        for target_name, meta_dict in self.metadata.items():
            outs_count: int = meta_dict["output_dimension"]
            if "profile" in target_name.lower():
                categorical_details.append(current_index)

            current_index += outs_count

        return categorical_details

    @property
    def get_categorical_output_info(self) -> dict[int, int]:
        """
        get the categorical metadata that we need for setting up the categorical funnel
        Returns
        -------
        dict of encoded_array_idx, number of categories
        """
        categorical_details = dict()
        current_index: int = 0
        for meta_dict in self.metadata.values():
            # tracker: int = meta_dict["idx"] + 1
            outs_count: int = meta_dict["output_dimension"]
            if meta_dict["enc_type"] == "categorical":
                categorical_details.update({current_index: meta_dict["num_categories"]})

            current_index += outs_count

        return categorical_details

    @property
    def get_numeric_output_info(self) -> list[int]:
        """
        Metadata on the position of numeric variables in the  output array
        Returns
        -------
        returns the indicies of the numeric variables in the encoded array
        """
        numeric_idxs = list()
        current_index: int = 0
        for meta_dict in self.metadata.values():
            # tracker: int = meta_dict["idx"] + 1
            outs_count: int = meta_dict["output_dimension"]
            if (meta_dict["enc_type"] == "numeric") or meta_dict["enc_type"] == "chronological":
                for n in range(outs_count):
                    numeric_idxs.append(current_index + n)

            current_index += outs_count

        return numeric_idxs

    @property
    def get_text_output_info(self) -> tuple[list[int], int]:
        """
        Metadata on the position of numeric variables in the  output array
        Returns
        -------
        the text indicies in the encoded array, and the maximum tokenized lengths
        """
        text_idxs = list()
        max_lengths = list()
        current_index: int = 0
        for meta_dict in self.metadata.values():
            outs_count: int = meta_dict["output_dimension"]
            if meta_dict["enc_type"] == "text":
                max_lengths.append(outs_count)
                for n in range(outs_count):
                    text_idxs.append(current_index + n)

            current_index += outs_count

        return text_idxs, sum(max_lengths)

    @property
    def output_column_types(self) -> list[str]:
        """
        Get the type of each output column (by index)
        Returns
        -------
        list of strings
        """
        # assumptions: that our encoders are in their expected / sorted order
        tracker: int = 0
        current_index: int = 0
        output_types: list = []

        for meta_dict in self.metadata.values():
            assert tracker >= meta_dict["idx"]
            tracker: int = meta_dict["idx"] + 1
            outs_count: int = meta_dict["output_dimension"]
            # variable_idx_range: tuple = (current_index, current_index + outs_count)
            encoder_type = meta_dict["enc_type"]

            output_types.extend([encoder_type] * outs_count)
            current_index += outs_count

        return output_types

    @property
    def is_fitted(self) -> None:
        tracker = []
        for name, encoder in self.encoders.items():
            if encoder.is_fitted:
                tracker.append(True)
                # print(f"{name} has been fitted")
            else:
                tracker.append(False)
                print(f"{name} HAS NOT BEEN FITTED")

        return np.all(tracker)

    def __repr__(self):
        return str(self.metadata)

    def __str__(self):
        return str(self.metadata)

    def __getitem__(self, key: str):
        return self.encoders.get(key, None)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        # Return a dictionary that defines what to pickle
        # we need to account for individual serialization
        state = self.__dict__.copy()  # Copy the current state
        if not state["frozen"]:
            raise ValueError("pipeline is not yet frozen...")

        state["encoders"] = OrderedDict()
        # defining this here in case we need specific behaviors to be included
        return state


def save_pipeline(
        pipeline: UniversalPipeline,
        base_path: str = "../../../data/",
        filename: str = "pipeline"
):
    """
    This would mean directories, packagelist, versioning on serialization
            -root/
            - pipeline object.pkl
            - versioning.txt (-Torch, numpy, sentencepiece dependencies)
            - packlist.txt
                - Encoders/
                    - encoders (.pkl each)

    Parameters
    ----------
    pipeline : the pipeline object - instance of UniversalPipeline
    filename : the filename/path to serialize the objects and root of the folder structure

    Returns
    -------
    None
    """
    folder_structure = {
        f"{filename}": {
            "documents": {},
            "encoders": {},
        },
    }
    # -----
    assert pipeline.frozen is True
    _base_path = base_path

    def build_folders(folder_structure: dict, _base_path=_base_path):
        for folder_name, subfolders in folder_structure.items():
            path = os.path.join(_base_path, folder_name)
            os.makedirs(path, exist_ok=True)
            print(f"built {path}")
            if subfolders:
                _base_path += f"{folder_name}"
                build_folders(subfolders, _base_path)

    build_folders(folder_structure, _base_path)
    packlist = [
        f"--------- Writing Pipeline object on {datetime.now()} ---------",
    ]

    # pickle each encoder object - each should have any save procedures in its __getstate__ method
    for enc_position, (enc_name, enc_obj) in enumerate(pipeline.encoders.items()):
        pickle_name = f"{base_path}{filename}encoders/{enc_position}_{enc_name}.pkl"
        packlist.append(f"---------\nwriting {pickle_name}\n{enc_obj.metadata}")
        with open(pickle_name, mode="wb") as f:
            pickle.dump(enc_obj, f, protocol=pickle.DEFAULT_PROTOCOL)  # 4


    pipeline.encoders = {}
    gc.collect()
    # serialize the pipeline "shell object" (without encoders or VAE)
    with open(f"{base_path}{filename}pipeline.pkl", mode="wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.DEFAULT_PROTOCOL)  # 4
    packlist.append(f"---------\n pipeline written to {base_path}{filename}pipeline.pkl")
    packlist.append(f"{pipeline.metadata}")

    packlist = "\n".join(packlist)
    with open(f"{base_path}{filename}documents/packlist.txt", "w") as f:
        f.write(packlist)

    versions = (
        # f"torch == {torch.__version__}\n"
        # f"sentencepiece {spm.__version__}\n"
        f"numpy {np.__version__}\n"
        f"pandas {pd.__version__}\n"
        f"scipy {scipy.__version__}"
        f"pickle {pickle.format_version} PROTOCOL {pickle.DEFAULT_PROTOCOL}\n"
        f"pipeline V0.0"  # todo: automate with packageversion
    )
    with open(f"{base_path}{filename}documents/versioning.txt", "w") as f:
        f.write(versions)


def load_pipeline(filename: str = "pipeline") -> UniversalPipeline:
    """
    Loading the pipeline, encoder objects and VAE imputer
    Parameters
    ----------
    filename : string of target pipeline folder

    Returns
    -------
    the loaded Universal Pipeline object
    """

    folder_structure = {
        f"{filename}": {
            "documents": {},
            "encoders": {"tokenizer": {}},
        },
    }
    # -----
    # TODO: add conditions for non-matching folder structure
    valid_folders = []
    for folder_name in folder_structure[f"{filename}"].keys():
        valid = os.path.exists(f"{filename}{folder_name}")
        valid_folders.append(valid)

    if not all(valid_folders):
        raise IOError("Issue with the expected folder structure")

    if os.path.exists(f"{filename}pipeline.pkl"):
        with open(f"{filename}pipeline.pkl", mode="rb") as f:
            pipeline = pickle.load(f)
    else:
        raise FileNotFoundError("pipeline file not found")


    encoder_files = os.scandir(f"{filename}encoders/")
    # unordered load here; but we'll sort encoders later by their target_idx
    encoder_files = [file.name for file in encoder_files if ".pkl" in file.name]

    for object_name in encoder_files:
        # validate the position of the encoders:
        if ".pkl" in object_name and object_name[0].isdigit():
            # eg 0_power.pkl, 1_proposal.pkl
            enc_num = object_name[0 : object_name.find("_")]  # first _ s
            enc_name = object_name[object_name.find("_") + 1 : object_name.rfind(".")]
            validated = pipeline._order[int(enc_num)] == enc_name
            if validated:
                with open(f"{filename}encoders/{object_name}", mode="rb") as f:
                    pipeline.encoders[enc_name] = pickle.load(f)

    pipeline._sort_encoders()
    return pipeline
