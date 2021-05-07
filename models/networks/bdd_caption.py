import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18

try:
    from .mlp import MLP
    from .factory_fusions import factory as factory_fusions
except:
    from mlp import MLP
    from factory_fusions import factory as factory_fusions

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, feature_dim, n_vocab=1290, max_length=20, gru_lstm="gru"):
        # max_length is the length of the input sequence to be attended
        # n_vocab is the vocab_size (=NULL + START + END + UNK + 1000 words)
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_vocab = n_vocab

        self.embedding = nn.Embedding(n_vocab, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)
        self.attn_combine = nn.Linear(hidden_size + feature_dim, hidden_size)
        self.gru_lstm = gru_lstm
        if self.gru_lstm == "gru":
            RNN = nn.GRU
        elif self.gru_lstm == "lstm":
            RNN = nn.LSTM
        else:
            raise NotImplementedError
        self.RNN = RNN(hidden_size, hidden_size)
        self.w_out = nn.Linear(hidden_size, n_vocab)

    def forward(self, inp, hidden, all_features):
        """
        inp is the previous word is made of prev
        hidden is the previous state of the RNN (or (h, c) for LSTM)
        all_features is made of all features on which attention is applied (batch X max_length X feature_dim)
        """

        # Embed previous word
        embedded = self.embedding(inp)

        # Concat previous word with previous hidden and compute attention weights
        # batch X max_length
        if self.gru_lstm == "gru":
            h = hidden
        elif self.gru_lstm == "lstm":
            h = hidden[0]
        else:
            assert False
        attn_weights = F.softmax(self.attn(torch.cat((embedded, h[0]), 1)), dim=1)

        # Apply attention weights to all_features
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), all_features).squeeze(1) # batch X feature_dim

        output = torch.cat((embedded, attn_applied), 1) # batch X (hidden_size + feature_dim)
        output = F.relu(self.attn_combine(output)) # 1 X batch X hidden_size

        # Apply RNN for one step
        # output is (1, batch X hidden_size)
        # if GRU: hidden is (1, batch X hidden_size)
        # if LSTM: hidden is ((1, batch X hidden_size), (1, batch X hidden_size))
        output, hidden = self.RNN(output.unsqueeze(0), hidden)

        # Get probability for next word
        # batch X n_vocab
        output = F.log_softmax(self.w_out(output.squeeze(0)), dim=1)

        return output, hidden


    def init_hidden(self, bsize):
        hidden = torch.zeros(1, bsize, self.hidden_size) # first dimension (1) represent a one-layer RNN
        if self.gru_lstm == "gru":
            return hidden
        # else this is a lstm
        cell = torch.zeros(1, bsize, self.hidden_size) # first dimension (1) represent a one-layer RNN
        return (hidden, cell)


class Captioning(nn.Module):
    def __init__(self,
                 hidden_size=500,
                 n_vocab=1290,
                 max_length=20,
                 decoded_sentence_max_length=30,
                 temperature=0.03,
                 sampling_strategy="top1",
                 fusion_opt={},
                 gru_lstm="gru",
                 output_sentence="caption",
                 ):
        super(Captioning, self).__init__()
        self.n_vocab = n_vocab
        self.decoded_sentence_max_length = decoded_sentence_max_length
        self.temperature = temperature
        self.sampling_strategy = sampling_strategy
        self.fusion_opt = fusion_opt
        self.layers_to_fuse = self.fusion_opt.pop("layers_to_fuse", None)
        self.gru_lstm = gru_lstm
        self.output_sentence = output_sentence # either action, justification or caption
        assert self.output_sentence in ["action", "justification", "caption"]

        # Provide dimensions
        self.layer_to_dim = {"stem": 64, "layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512, "output": 512, "prediction": 2}
        self.fusion_opt["input_dims"] = []
        for layer in self.layers_to_fuse:
            self.fusion_opt["input_dims"].append(self.layer_to_dim[layer])

        if len(self.layers_to_fuse) == 1:
            self.feature_dim = self.layer_to_dim[self.layers_to_fuse[0]]
        elif len(self.layers_to_fuse) == 2:
            self.feature_dim = fusion_opt["output_dim"]
            self.fusion = factory_fusions(self.fusion_opt)
        else:
            assert False

        # feature_dim = feature_dim+2 as we concatenate predicted acceleration and change of course
        self.decoder = AttnDecoderRNN(hidden_size=hidden_size, feature_dim=self.feature_dim, n_vocab=n_vocab, max_length=max_length, gru_lstm=self.gru_lstm)

    def forward(self, batch):
        input_features = []
        for layer in self.layers_to_fuse:
            input_features.append(batch[layer]) # batch X max_length=20 X input_dim

        caption = batch[self.output_sentence] # batch X n_words=22 (=20 words + start + sep)
        mask = batch["mask"] # Batch X max_length=20
        bsize = caption.size(0)
        n_words = caption.size(1)

        # batch X time=20 X self.fusion_opt["output_dim"]
        if len(input_features) > 1:
            # Flatten the first two dimensions
            flattened_input_features = []
            for input_feature in input_features:
                flattened_input_features.append(input_feature.view(-1, input_feature.size(2)))
            all_features = self.fusion(flattened_input_features)
            # Reshape to 3D tensor
            all_features = all_features.view(bsize, mask.size(1), -1)
        else:
            all_features = input_features[0]

        # Apply mask (element-wise multiplication)
        all_features = all_features * mask.unsqueeze(2)

        hidden = self.decoder.init_hidden(bsize=bsize)
        if caption.is_cuda:
            if self.gru_lstm == "gru":
                hidden = hidden.cuda()
            else:
                hidden = (hidden[0].cuda(), hidden[1].cuda())

        outputs = []
        for i in range(n_words - 1):
            output, hidden = self.decoder(inp=caption[:, i], hidden=hidden, all_features=all_features) # teacher forcing
            # output: (batch X n_vocab)
            outputs.append(output.unsqueeze(0))

        outputs = torch.cat(outputs, 0)
        outputs = outputs.transpose(0, 1) # batch X (n_words-1) X n_vocab
        net_out = {"predicted_sentence_proba": outputs}

        # Generate tokens (and not only probabilities)
        hidden = self.decoder.init_hidden(bsize=bsize)
        if self.gru_lstm == "gru":
            hidden = hidden.cuda()
        else:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        decoder_input = caption[:, 0]
        decoded_sentence = []
        for i in range(self.decoded_sentence_max_length):
            output, hidden = self.decoder(inp=decoder_input, hidden=hidden, all_features=all_features) # teacher forcing

            # Either take top 1 or sample with probability and temperature
            if self.sampling_strategy == "top1":
                _, token = output.topk(1)
            elif self.sampling_strategy == "sample":
                token =  torch.multinomial(output.div(self.temperature).exp(), 1)

            token = token.detach()
            decoded_sentence.append(token)
            decoder_input = token[:, 0]

        net_out["sentence_decoded"] = torch.cat(decoded_sentence, dim=1).cpu()

        return net_out

