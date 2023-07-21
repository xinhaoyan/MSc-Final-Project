# Literature Review 

## The first paper tital  is “ NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE ”

The paper describes a new approach to machine translation called Neural Machine Translation (NMT). NMT uses a single neural network to maximize translation performance and allows the model to automatically search for relevant parts of a source sentence. This approach improves translation performance and achieves results comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. 

More important thing is that this paper introduces a Recurrent Neural Network (RNN) attention mechanism that improves the model's remote sequence modelling capabilities. This allows the RNN to translate longer sentences more accurately - a motivation for the later development of the original Transformer architecture.

URL of the paper：https://arxiv.org/pdf/1409.0473.pdf

##  The second paper tital is "Attention Is All You Need"

This paper introduces the Transformer network, a new architecture for sequence transduction models based solely on attention mechanisms. The attention mechanism offers advantages in terms of model performance, training time, and parallelization. The Transformer network differs from traditional sequence transduction models based on recurrent or convolutional neural networks in that it does not rely on sequential processing, making it more efficient and easier to parallelize. The Transformer network has been successfully applied to a variety of tasks beyond machine translation, including language modeling, text generation, and speech recognition.

This paper introduces the original Transformer architecture consisting of encoders and decoders, which will be related later as separate modules. In addition, this paper introduces the concepts of scaled dot product attention mechanisms, multi-head attention blocks, and positional input coding, which remain the foundations of the modern Transformer.

URL of the paper：https://arxiv.org/abs/1706.03762
