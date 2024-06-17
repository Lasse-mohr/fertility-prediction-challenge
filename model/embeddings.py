import torch
import torch.nn as nn


class SurveyEmbeddings(nn.Module):
    def __init__(self, vocab_size: int,
                 n_questions: int,
                 n_years: int = 14,
                 embedding_dim: int = 16,
                 dropout: float = None):
        super().__init__()
        # Vocab size is number of unique answers
        self.answer_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=101)
        self.answer_embedding_cont = nn.Linear(1, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.yearly_embedding = nn.Embedding(
            n_years, embedding_dim)  # 14 years of data
        self.question_embedding = nn.Embedding(
            n_questions, embedding_dim)  # Number of unique questions
        self.register_buffer(
            "question_range", torch.arange(n_questions))  # )

        self.register_parameter("alpha", nn.Parameter(
            torch.tensor([0.01]), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(
            torch.tensor([0.2]), requires_grad=True))

        self.reset_parameters()

        if dropout is not None:
            self.drop_year = nn.Dropout1d(dropout)
            self.drop_answer = None  # nn.Dropout1d(dropout)
            self.drop_question = None  # nn.Dropout1d(dropout)

        self.return_status()
        self.embedding_dim = embedding_dim

    def reset_parameters(self):
        # , a=-0.5, b=0.5)
        nn.init.kaiming_normal_(self.answer_embedding.weight)
        # , a=-0.5, b=0.5)
        nn.init.kaiming_normal_(self.yearly_embedding.weight)
        nn.init.kaiming_normal_(self.question_embedding.weight)

    def return_status(self):
        if self.drop_answer is not None:
            print("Embedding Layer with the Dropout")

    def forward(self, year, answer):
        # ANSWER EMBEDDING
        placeholder = torch.zeros(*answer.shape, self.embedding_dim)
        is_continious = answer <= 1
        embed_continious = self.answer_embedding_cont(answer[is_continious].unsqueeze(1))
        embed_cat = self.answer_embedding(answer[~is_continious].unsqueeze(1))

        placeholder[is_continious] = embed_continious.squeeze(1)
        placeholder[~is_continious] = embed_cat.squeeze(1)
        answer = self.norm(placeholder)

        # answer = self.answer_embedding(answer)
        if self.drop_answer is not None:
            answer = self.drop_answer(answer)
        # YEAR EMBEDDING
        year = self.yearly_embedding(year).unsqueeze(1)
        if self.drop_year is not None:
            year = self.drop_year(year)
        embeddings = answer + self.alpha * year
        # QUESTION EMBEDDING
        question = self.question_embedding(self.question_range)
        if self.drop_question is not None:
            question = self.drop_question(question)
        embeddings = embeddings + self.beta * question

        return embeddings
