"""
Test cases to ensure native Forte code can be imported
with only backbone Forte library installed (without any
extra imports).
"""


def test_basic_import():
    from forte.data import DataPack
    from forte.data import DataStore
    from forte.data import SinglePackSelector
    from forte.data import BaseIndex


def test_import_data():
    from forte.data.readers import (
        AGNewsReader,
        ClassificationDatasetReader,
        CoNLL03Reader,
        ConllUDReader,
        RawDataDeserializeReader,
        RecursiveDirectoryDeserializeReader,
        DirPackReader,
        MultiPackDirectoryReader,
        MultiPackDeserializerBase,
        SinglePackReader,
        HTMLReader,
        LargeMovieReader,
        TerminalReader,
        RawPackReader,
        RawMultiPackReader,
        MSMarcoPassageReader,
        MultiPackSentenceReader,
        MultiPackTerminalReader,
        OntonotesReader,
        OpenIEReader,
        PlainTextReader,
        ProdigyReader,
        RACEMultiChoiceQAReader,
        SemEvalTask8Reader,
        SST2Reader,
        StringReader,
    )
    from forte.datasets.mrc.squad_reader import SquadReader


def test_import_processors():
    from forte.processors.writers import (
        PackIdJsonPackWriter,
    )
    from forte.processors.nlp import (
        ElizaProcessor,
    )
    from forte.processors.misc import (
        AnnotationRemover,
        AttributeMasker,
        DeleteOverlapEntry,
        LowerCaserProcessor,
        PeriodSentenceSplitter,
        WhiteSpaceTokenizer,
        Alphabet,
        VocabularyProcessor,
    )
    from forte.processors.base import (
        BaseProcessor,
    )
    from forte.processors.data_augment import (
        BaseDataAugmentProcessor,
    )
    from forte.processors.ir.search_processor import (
        SearchProcessor,
    )


def test_import_evaluator():
    from forte.evaluation.ner_evaluator import (
        CoNLLNEREvaluator,
    )
    from forte.evaluation.base import Evaluator


def test_import_trainer():
    from forte.trainer.base import BaseTrainer


def test_import_forte_modules():
    from forte.pipeline_component import PipelineComponent
    from forte import Pipeline
    from forte.process_job import ProcessJob, ProcessJobStatus
    from forte.process_manager import ProcessManager
    from forte.train_pipeline import TrainPipeline


def test_import_base_data_aug():
    from forte.processors.data_augment import (
        BaseDataAugmentProcessor,
        ReplacementDataAugmentProcessor,
    )
    from forte.processors.data_augment.base_op_processor import (
        BaseOpProcessor,
    )

    from forte.processors.data_augment.algorithms.back_translation_op import (
        BackTranslationOp,
    )

    from forte.processors.data_augment.algorithms.back_translation_op import (
        BackTranslationOp,
    )
    from forte.processors.data_augment.algorithms.base_data_augmentation_op import (
        BaseDataAugmentationOp,
    )
    from forte.processors.data_augment.algorithms.character_flip_op import (
        CharacterFlipOp,
    )
    from forte.processors.data_augment.algorithms.dictionary_replacement_op import (
        DictionaryReplacementOp,
    )
    from forte.processors.data_augment.algorithms.dictionary import (
        Dictionary,
    )
    from forte.processors.data_augment.algorithms.distribution_replacement_op import (
        DistributionReplacementOp,
    )
    from forte.processors.data_augment.algorithms.eda_ops import (
        RandomSwapDataAugmentOp,
        RandomInsertionDataAugmentOp,
        RandomDeletionDataAugmentOp,
    )
    from forte.processors.data_augment.algorithms.sampler import (
        Sampler,
        UniformSampler,
        UnigramSampler,
    )

    from forte.processors.data_augment.algorithms.single_annotation_op import (
        SingleAnnotationAugmentOp,
    )
    from forte.processors.data_augment.algorithms.text_replacement_op import (
        TextReplacementOp,
    )
    from forte.processors.data_augment.algorithms.typo_replacement_op import (
        UniformTypoGenerator,
        TypoReplacementOp,
    )
    from forte.processors.data_augment.algorithms.word_splitting_op import (
        RandomWordSplitDataAugmentOp,
    )


if __name__ == "__main__":
    test_basic_import()
    test_import_data()
    test_import_processors()
    test_import_evaluator()
    test_import_trainer()
    test_import_forte_modules()
    test_import_base_data_aug()
