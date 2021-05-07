import os

from pathlib import Path

import torch
import torch.nn as nn

from bootstrap.lib.logger import Logger
from bootstrap.lib.options import Options

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


class CaptionMetrics(nn.Module):

    def __init__(self, decode_fn, bleu_smoothing=0, engine=None, mode=None, output_sentence="caption"):
        super(CaptionMetrics, self).__init__()
        self.decode_fn = decode_fn
        self.bleu_smoothing = bleu_smoothing
        self.engine = engine
        self.mode = mode
        self.print_count = 0
        self.sep_token = " <sep> "
        self.output_sentence = output_sentence # action, justification or caption

        self.sentences = {"ground_truth": {"action": [], "justification": []},
                          "predicted": {"action": [], "justification": []}}

        self.meta_info = {}

        if self.engine is not None:
            self.engine.register_hook(
                '%s_on_start_epoch'%mode,
                self.start_epoch)
            self.engine.register_hook(
                '%s_on_end_epoch'%mode,
                self.end_epoch)

    def start_epoch(self):
        N = len(self.engine.dataset[self.mode])
        self.print_count = 0
        self.print_every = N // 10

    def end_epoch(self,):
        path = Path(Options()["exp.dir"])

        dirname = path.joinpath("generated_sentences")
        # Create directory if it does not exist
        if not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Dump sentences to the directory
        for field in ["action", "justification"]:
            for key in ["ground_truth", "predicted"]:
                filepath = dirname.joinpath("%s_%s.txt" % (key, field))
                with open(filepath, "w") as f:
                    f.write("\n".join(self.sentences[key][field]))

        # Compute NLP quality scores (bleu, meteor, cider...)
        for field in ["action", "justification"]:
            cider = Cider()
            bleu = Bleu()
            meteor = Meteor()

            # Check if this is not empty
            if len(self.sentences["ground_truth"][field]) > 0:
                ground_truth = {i: [sentence] for i, sentence in enumerate(self.sentences["ground_truth"][field])}
                predicted = {i: [sentence] for i, sentence in enumerate(self.sentences["predicted"][field])}

                cider_score, _ = cider.compute_score(ground_truth, predicted)
                cider_score = cider_score * 100 # Convert to percentage

                bleus_score, _ = bleu.compute_score(ground_truth, predicted)
                bleu_score = bleus_score[3] * 100 # Take bleu-4 and convert to percentage

                meteor_score, _ = meteor.compute_score(ground_truth, predicted)
                meteor_score = meteor_score * 100 # Convert to percentage
            else:
                # Otherwise all scores are 0
                cider_score, bleu_score, meteor_score = 0, 0, 0

            Logger().log_value('%s_epoch.cider_%s' % (self.mode, field), cider_score, should_print=True)
            Logger().log_value('%s_epoch.bleucoco_%s' % (self.mode, field), bleu_score, should_print=True)
            Logger().log_value('%s_epoch.meteorcoco_%s' % (self.mode, field), meteor_score, should_print=True)

        # Reset sentences
        self.sentences = {"ground_truth": {"action": [], "justification": []},
                         "predicted": {"action": [], "justification": []}}
        return

    def forward(self, cri_out, net_out, batch):
        predictions = net_out["sentence_decoded"]
        gt_sentences = batch["%s_text" % self.output_sentence]

        for sentence, gt_sentence in zip(predictions, gt_sentences): # Iterate over the batch
            # Convert sentence tokens to sentence words
            decoded_sentence_words = self.decode_fn(sentence)

            # Remove end of sentences
            k = len(decoded_sentence_words) - 1
            while k >= 0 and decoded_sentence_words[k] in ["<NULL>", "<END>"]:
                k -= 1
            decoded_sentence_words = decoded_sentence_words[:k+1]

            decoded_sentence = " ".join(decoded_sentence_words)

            predicted = {}
            gt = {}
            # If self.output_sentence is "caption", split on the <sep> token
            if self.output_sentence == "caption":
                decoded_sentence_splitted = decoded_sentence.split(self.sep_token)
                # If no <sep> token or more than two
                if len(decoded_sentence_splitted) != 2:
                    continue

                predicted["action"], predicted["justification"] = decoded_sentence_splitted
                gt["action"], gt["justification"] = gt_sentence.split(self.sep_token)
            else:
                predicted[self.output_sentence] = decoded_sentence
                gt[self.output_sentence] = gt_sentence

            # Append to the full list of sentences
            for field in gt.keys():
                # Save everything for end_epoch computation
                self.sentences["ground_truth"][field].append(gt[field])
                self.sentences["predicted"][field].append(predicted[field])

            # Log stuff
            self.print_count += 1
            if not self.print_count % self.print_every:
                Logger()("GT   = %s" % gt_sentence)
                Logger()("Pred = %s" % decoded_sentence)

        return {}

