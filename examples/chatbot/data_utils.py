import random
import logging


def split_train_eval_test(file_name):
    r"""Split the dataset into train, eval and test

    Args:
        file_name (str): File path to the dataset
    """

    text_file = open(file_name, "r")
    text_data = []
    conversation = None

    for line in text_file:
        pair = line.rstrip('\n').split('\t')
        num, question = pair[0].split(' ', 1)
        answer = pair[1]

        if num == "1":
            if conversation:
                text_data.append(conversation)
            conversation = [[question, answer]]
        else:
            conversation.append([question, answer])

    text_data.append(conversation)

    text_file.close()

    logging.info("Total number of conversation is %s.", len(text_data))

    num_train = 8938
    num_val = 500
    num_test = 500

    train_data = text_data[0:num_train]
    val_data = text_data[num_train:num_train + num_val]
    test_data = text_data[num_train + num_val:num_train + num_val + num_test]

    return train_data, val_data, test_data


def generate_negative_examples(conversations):
    r"""Generate negative by shuffling the responses.

    Args:
         conversations (list): A list of `[context, response]` pairs.
    """

    contexts, responses = zip(*conversations)

    # negative responses are basically shuffled responses
    negative_response = list(responses)
    random.shuffle(negative_response)

    return contexts, responses, negative_response


def _create_conv_with_history(conv, num_qa):
    r"""Generate data using historical context for a single conversation.
    Here, by conversation we mean a sequence of turns between two entities A
    and B.

    Args:
        conv (list): A list of `[context, response]` pairs.
        num_qa (int): Length of historical context to be considered. For e.g.,
            `num_qa = 2`, would mean sentences from last two turns are
            appended to the current conversation.

    Returns:
        A list containing `[context, response]` pairs.
    """

    new_text_data = []
    for i, _ in enumerate(conv):
        history = conv[max(i - num_qa, 0):i]
        current_qa = conv[i]

        if history:
            qa_with_history = [sentence for qa in history for sentence in
                               qa] + current_qa
            qa_with_history = [' '.join(qa_with_history[:-1]),
                               qa_with_history[-1]]
        else:
            qa_with_history = current_qa

        new_text_data.append(qa_with_history)
    return new_text_data


def create_dataset_with_history(conversations, num_line=2):
    r"""Generate conversation data using historical conversation for entire
    dataset.

    Args:
        conversations (list): A list of conversations. Each conversation is in
            turn represented as a list of `[sent1, sent2]` pairs.
        num_line (int): Length of historical context to be considered. For e.g.,
            `num_line = 2`, would mean sentences from last two turns are
            appended to the current conversation.
    """

    logging.info("Total number of conversation is %s.", len(conversations))

    proc_text_data = []

    for conversation in conversations:
        conversation_with_history = _create_conv_with_history(
            conversation, num_line)
        proc_text_data.extend(conversation_with_history)

    return proc_text_data
