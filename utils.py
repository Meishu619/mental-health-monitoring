import audobject
import numpy as np


class LabelEncoder(audobject.Object):
    def __init__(self, labels):
        self.labels = sorted(labels)
        codes = range(len(labels))
        self.inverse_map = {code: label for code,
                    label in zip(codes, self.labels)}
        self.map = {label: code for code,
                            label in zip(codes, self.labels)}

    def encode(self, x):
        return self.map[x]

    def decode(self, x):
        return self.inverse_map[x]


class Standardizer(audobject.Object):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.tolist()
        self.std = std.tolist()
        self._mean = mean
        self._std = std
    
    def encode(self, x):
        return (x - self._mean) / (self._std)

    def decode(self, x):
        return x * self._std + self._mean

    def __call__(self, x):
        return self.encode(x)

# def transfer_features(features, device):
#     return features.to(device).float()


# def evaluate_regression(model, device, loader, transfer_func):
#     metrics = {
#         'CC': audmetric.pearson_cc,
#         'CCC': audmetric.concordance_cc,
#         'MSE': audmetric.mean_squared_error,
#         'MAE': audmetric.mean_absolute_error
#     }
#     model.to(device)
#     model.eval()
#     outputs = torch.zeros(len(loader.dataset))
#     targets = torch.zeros(len(loader.dataset))
#     with torch.no_grad():
#         for index, (features, target) in audeer.progress_bar(
#             enumerate(loader),
#             desc='Batch',
#             total=len(loader),
#             disable=True
#         ):
#             start_index = index * loader.batch_size
#             end_index = (index + 1) * loader.batch_size
#             if end_index > len(loader.dataset):
#                 end_index = len(loader.dataset)
#             outputs[start_index:end_index] = model(
#                 transfer_func(features, device)).squeeze()
#             targets[start_index:end_index] = target
#     targets = targets.numpy()
#     outputs = outputs.numpy()
#     return {
#         key: metrics[key](targets, outputs)
#         for key in metrics.keys()
#     }, targets, outputs


# def evaluate_classification(model, device, loader, transfer_func):
#     metrics = {
#         'UAR': audmetric.unweighted_average_recall,
#         'ACC': audmetric.accuracy,
#         'F1': audmetric.unweighted_average_fscore
#     }
#     model.to(device)
#     model.eval()

#     outputs = torch.zeros((len(loader.dataset)))
#     targets = torch.zeros(len(loader.dataset))
#     with torch.no_grad():
#         for index, (features, target) in audeer.progress_bar(
#             enumerate(loader),
#             desc='Batch',
#             total=len(loader),
#             disable=True
#         ):
#             start_index = index * loader.batch_size
#             end_index = (index + 1) * loader.batch_size
#             if end_index > len(loader.dataset):
#                 end_index = len(loader.dataset)
#             outputs[start_index:end_index] = model(
#                 transfer_func(features, device)).argmax(dim=1)
#             targets[start_index:end_index] = target
#     targets = targets.numpy()
#     outputs = outputs.numpy()
#     return {
#         key: metrics[key](targets, outputs)
#         for key in metrics.keys()
#     }, targets, outputs


# class CCCLoss(torch.nn.Module):
#     def forward(self, output, target):
#         out_mean = torch.mean(output)
#         target_mean = torch.mean(target)

#         covariance = torch.mean((output - out_mean) * (target - target_mean))
#         target_var = torch.mean((target - target_mean)**2)
#         out_var = torch.mean((output - out_mean)**2)

#         ccc = 2.0 * covariance / \
#             (target_var + out_var + (target_mean - out_mean)**2 + 1e-10)
#         loss_ccc = 1.0 - ccc

#         return loss_ccc


# class MyCCC(torch.nn.Module):
#     def forward(self, output, target):
#         out_mean = torch.mean(output)
#         target_mean = torch.mean(target)

#         covariance = torch.mean((output - out_mean) * (target - target_mean))
#         target_var = torch.mean((target - target_mean)**2)
#         out_var = torch.mean((output - out_mean)**2)

#         ccc = 2.0 * covariance / \
#             (target_var + out_var + (target_mean - out_mean)**2 + 1e-10)


#         return ccc



# class MinMaxScaler(audobject.Object):
#     def __init__(self, minimum: float, maximum: float):
#         self.minimum = float(minimum)
#         self.maximum = float(maximum)

#     def encode(self, x):
#         return (x - self.minimum) / (self.maximum - self.minimum)

#     def decode(self, x):
#         return x * (self.maximum - self.minimum) + self.minimum


# def evaluate_multitask(
#     model,
#     device, 
#     loader,
#     task_dict,
#     transfer_func,
#     output_dim: int = None,
#     score: bool = True
# ):
#     metrics = {
#         'classification': {
#             'UAR': audmetric.unweighted_average_recall,
#             'ACC': audmetric.accuracy,
#             'F1': audmetric.unweighted_average_fscore
#         },
#         'regression': {
#             'CC': audmetric.pearson_cc,
#             'CCC': audmetric.concordance_cc,
#             'MSE': audmetric.mean_squared_error,
#             'MAE': audmetric.mean_absolute_error
#         }
#     }

#     model.to(device)
#     model.eval()

#     outputs = torch.zeros((len(loader.dataset), model.output_dim if output_dim is None else output_dim))
#     if score:
#         targets = torch.zeros((len(loader.dataset), len(task_dict)))
#     with torch.no_grad():
#         for index, (features, target) in tqdm.tqdm(
#             enumerate(loader),
#             desc='Batch',
#             total=len(loader),
#             disable=score
#         ):
#             start_index = index * loader.batch_size
#             end_index = (index + 1) * loader.batch_size
#             if end_index > len(loader.dataset):
#                 end_index = len(loader.dataset)
#             outputs[start_index:end_index, :] = model(
#                 transfer_func(features, device))
#             if score:
#                 targets[start_index:end_index] = target
#             # break

#     outputs = outputs.cpu().numpy()
#     if not score:
#         return outputs
#     targets = targets.numpy()
#     predictions = []
#     results = {}
#     for task in task_dict:
#         results[task] = {}
#         if task_dict[task]['type'] == 'regression':
#             preds = outputs[:, task_dict[task]['unit']]
#         else:
#             preds = outputs[:, task_dict[task]['unit']].argmax(1)
#         predictions.append(preds)
#         for metric in metrics[task_dict[task]['type']]:
#             results[task][metric] = metrics[task_dict[task]['type']][metric](
#                 targets[:, task_dict[task]['target']],
#                 preds
#             )
#     predictions = np.stack(predictions).T
#     total_score = []
#     for task in task_dict:
#         score = results[task][task_dict[task]['score']]
#         if task_dict[task]['score'] in ['MAE', 'MSE']:
#             score = 1 / (score + 1e-9)
#         total_score.append(score)
#     emo_score = sum([x for x, y in zip(total_score, task_dict) if y in EMOTIONS]) / len(EMOTIONS)
#     if len(task_dict) == len(EMOTIONS):
#         total_score = emo_score
#     elif len(task_dict) == 1:
#         total_score = total_score[0]
#     else:
#         scores = [emo_score] + [x for x, y in zip(total_score, task_dict) if y not in EMOTIONS]
#         total_score = len(scores) / sum([1 / (score + 1e-9) for score in scores])

#     return total_score, results, targets, outputs, predictions


# if __name__ == '__main__':
#     m = MinMaxScaler(0, 10)
#     print(m.encode(10))
#     print(m.encode(0))
#     print(m.encode(2))
#     print(m.encode(5))

#     print(m.decode(m.encode(10)))
#     print(m.decode(m.encode(0)))
#     print(m.decode(m.encode(2)))
#     print(m.decode(m.encode(5)))