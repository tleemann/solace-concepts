## Implement the coherence assessment modules.
import torchvision.models as models
import torch
# Best-Aligned Meaning (BAM)
class BestAlignedMeaningAssessor:
    def assess(self, meanings, concepts=None) -> float:
        """ Meaning: List of JSON dicts with keys {word, score} """
        scores = [item["score"] for item in meanings]
        return max(scores)


# Visual Embeddings Assessor.
## Some helpers

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def pairwise_cos_matrix(image_features):
    similarity = image_features @ image_features.t()
    return similarity

def pairwise_eucl_matrix(image_features):
    vnorms2 = torch.sum(torch.pow(image_features, 2.0), dim=1)
    # print(vnorms2.shape)
    similarity = vnorms2.reshape(-1,1) - 2.0*(image_features @ image_features.t()) + vnorms2.reshape(1,-1)
    #print(similarity)
    return similarity

def min_similarity(image_features, distance_fn):
    return torch.min(distance_fn(image_features))

def mean_similarity(image_features, distance_fn):
    return torch.mean((distance_fn(image_features)).type(torch.FloatTensor))

class VisualEmbeddingsAssessor():
    def __init__(self, device="cuda",  target_layer = 'layer4.2', coherence_measure="mean", cam_weighted=True, model=None, modelclass = models.resnet50):
        """
        :param coherence_measure: "mean" or "min" for mean cos distance between embeddings, f_ve,mean, 
            or min cos-distances between embeddings, f_ve,min (see paper for details.)  
        :param cam_weighted: Weigh features by the concept activation maps.
        """
        self.device = device
        if model is None:
            self.mymodel = modelclass(pretrained=True).to(self.device)
        else:
            self.mymodel = model 
        self.target_layer = target_layer
        self.mymodel.eval()
        self.weighted = cam_weighted
        self.coherence_measure = coherence_measure
    
    def assess(self, meanings, concepts) -> float:
        """ Meaning: List of JSON dicts with keys {word, score} """
        # Compute the input concepts' feature vectors.

        tensorlist = []
        saliency_list = []
        for example in concepts:
            tensorlist.append(example.image_tensor.expand(3, 448, 448)) # shape 3, 448, 448
            saliency_list.append(example.saliency_map)
        input_img = torch.stack(tensorlist)
        # Normalization prior to model.
        input_img = (input_img - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        concept_map = torch.stack(saliency_list).to(self.device)

        # Now have the batch of examples and concept activation maps ready.
        with torch.no_grad():            
            self.mymodel.layer4.register_forward_hook(get_activation(self.target_layer))
            image_features = self.mymodel(input_img.to(self.device))
            #print(input_img.shape)
            # image_features = image_features / image_features.norm(p=2, dim=1,keepdim=True)
            feature_map = activation[self.target_layer]
            #print(feature_map.shape)

            # Normalize spatial features to one over channels.
            feature_map = feature_map / (torch.norm(feature_map, p=2, dim=1, keepdim=True)+1e-5)

            if self.weighted:
                # Normalizate the concept_activation maps so they sum up to one
                concept_map = concept_map/torch.sum(concept_map.reshape(len(concept_map),-1), axis=1).reshape(len(concept_map), 1, 1)

                weighted_feature = feature_map*concept_map.unsqueeze(1)
                weighted_feature = torch.sum(torch.sum(weighted_feature,dim=-1), dim=-1)
                feature_vectors = weighted_feature / (torch.norm(weighted_feature, p=2, dim=1, keepdim=True)+1e-5)
            else:
                feature_map_v = torch.sum(torch.sum(feature_map, dim=-1), dim=-1) # Sum up feature maps for each channel.
                feature_vectors = feature_map_v / (torch.norm(feature_map_v, p=2, dim=1, keepdim=True)+1e-5) # Normalize each feature vector to one.

            # feature vectors.shape now is [num exemplars x num features]

            if self.coherence_measure == "mean":
                return mean_similarity(feature_vectors, pairwise_cos_matrix).item()
            elif self.coherence_measure == "min":
                return min_similarity(feature_vectors, pairwise_cos_matrix).item()
            else:
                raise ValueError("Unknown coherence measure.")

class HybridAssessor():
    """ The weighted hybrid assessor. """
    def __init__(self, assessor_list, weights_list):
        self.assessor_list = assessor_list
        self.weights_list = weights_list

    def assess(self, meanings, concepts) -> float:
        ret = 0.0
        for assessor, weight in zip(self.assessor_list, self.weights_list):
            ret += weight*assessor.assess(meanings, concepts)
        return ret