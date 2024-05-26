import torch
import torch.nn as nn


class SurveyEmbeddings(nn.Module):
    def __init__(self, vocab_size: int,
                 n_questions: int = None,
                 n_years: int = 14,
                 embedding_dim: int = 16,
                 dropout: float = None):
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

        self.register_parameter("alpha", nn.Parameter(
            torch.tensor([0.0]), requires_grad=True))
        self.register_parameter("beta", nn.Parameter(
            torch.tensor([0.2]), requires_grad=True))

        self.reset_parameters()

        if dropout is not None:
            self.drop_year = nn.Dropout(dropout)
            self.drop_answer = nn.Dropout(dropout)
            self.drop_question = nn.Dropout(dropout)

        self.return_status()

    def reset_parameters(self):
        nn.init.uniform_(self.answer_embedding.weight, a=-0.1, b=0.1)
        nn.init.uniform_(self.yearly_embedding.weight, a=-0.1, b=0.1)
        nn.init.orthogonal_(self.question_embedding.weight)

    def return_status(self):
        if self.question_embedding is not None:
            print("Embedding Layer with Question Embdeddings")
        if self.drop_answer is not None:
            print("Embedding Layer with the Dropout")

    def forward(self, year, answer):
        # ANSWER EMBEDDING
        answer = self.answer_embedding(answer)
        if self.drop_answer is not None:
            answer = self.drop_answer(answer)
        # YEAR EMBEDDING
        year = self.yearly_embedding(year).unsqueeze(1)
        if self.drop_year is not None:
            year = self.drop_year(year)
        embeddings = answer + self.alpha * year
        # QUESTION EMBEDDING
        if self.question_embedding is not None:
            question = self.question_embedding(self.question_range)
            if self.drop_question is not None:
                question = self.drop_question(question)
            embeddings = embeddings + self.beta * question

        return embeddings
