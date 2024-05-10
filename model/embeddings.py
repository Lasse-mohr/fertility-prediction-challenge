import torch


class SurveyEmbeddings(torch.nn.Module):
    def __init__(self, vocab_size: int, n_questions: int = None, n_years: int = 14, embedding_dim: int = 16):
        super().__init__()
        self.answer_embedding = torch.nn.Embedding(vocab_size, embedding_dim) # Vocab size is number of unique answers
        self.yearly_embedding = torch.nn.Embedding(n_years, embedding_dim) # 14 years of data
        if n_questions is not None:
            self.question_embedding = torch.nn.Embedding(n_questions, embedding_dim) # Number of unique questions
            self.question_range = torch.arange(n_questions) # Fixed range of questions

    def forward(self, answer, year):
        answer = self.answer_embedding(answer)
        year = self.yearly_embedding(year)
        embeddings = answer + year
        if hasattr(self, 'question_embedding'):
            embeddings += self.question_embedding(self.question_range)
        return embeddings