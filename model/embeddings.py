import torch
import torch.nn as nn


class SurveyEmbeddings(nn.Module):
    def __init__(self, vocab_size: int,
                 n_questions: int = None,
                 n_years: int = 14,
                 embedding_dim: int = 16):
        super().__init__()
        # Vocab size is number of unique answers
        self.answer_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=101)
        self.yearly_embedding = nn.Embedding(
            n_years, embedding_dim)  # 14 years of data
        if n_questions is not None:
            self.question_embedding = nn.Embedding(
                n_questions, embedding_dim)  # Number of unique questions
            self.register_buffer(
                "question_range", torch.arange(n_questions))  # )
            # self.question_range = torch.arange(n_questions) # Fixed range of questions
        else:
            self.question_embedding = None

        if self.question_embedding is not None:
            print("Embedding Layer with Question Embdeddings")

        self.register_parameter("alpha", nn.Parameter(
            torch.tensor([0.0]), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(
            torch.tensor([0.2]), requires_grad=True))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.answer_embedding.weight, a=-0.5, b=0.5)
        nn.init.uniform_(self.yearly_embedding.weight, a=-0.1, b=0.1)
        nn.init.orthogonal(self.question_embedding.weight)

    def forward(self, year, answer):
        answer = self.answer_embedding(answer)
        year = self.yearly_embedding(year)
        embeddings = answer + self.alpha * year.unsqueeze(1)
        if self.question_embedding is not None:
            embeddings = embeddings + self.beta * \
                self.question_embedding(self.question_range)
        return embeddings
