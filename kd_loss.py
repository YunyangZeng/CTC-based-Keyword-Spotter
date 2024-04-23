import torch




class KnowledgeDistillationLoss(torch.nn.Module):
    def __init__(self, loss_type='mse', temperature=1.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.criterion = torch.nn.MSELoss()
            self.temperature = None
        elif self.loss_type == 'kld':
            self.criterion = torch.nn.KLDivLoss(reduction='batchmean')
            self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        if self.loss_type == 'mse':
            return self.mse_loss(student_logits, teacher_logits)
        elif self.loss_type == 'kld':
            return self.kld_Loss(student_logits, teacher_logits)
    def mse_loss(self, student_logits, teacher_logits):
        return self.criterion(student_logits, teacher_logits)
    def kld_Loss(self, student_logits, teacher_logits):
        student_log_softmax = torch.nn.functional.log_softmax(student_logits/self.temperature, dim=1)
        teacher_softmax = torch.nn.functional.softmax(teacher_logits/self.temperature, dim=1)

        kld_loss = self.criterion(student_log_softmax, teacher_softmax) * (self.temperature ** 2)
        return kld_loss






