import json
import numpy as np
from collections import defaultdict

class Data(object):
    def __init__(self, data_filename):
        with open(data_filename, 'r') as f:
            data_json = json.load(f)
        self.data_json = data_json
        self.preprocess_data()

    def unroll_instance_dict(self, instance_dict):
        """ 
        Replace array-valued variables with unrolled names. This needs to match the
        compiler's policy for naming unrolled array elements.
        """
        new_dict = {}
        for var_name, vals in instance_dict.iteritems():
            if isinstance(vals, list):
                for i, val in enumerate(vals):
                    new_dict["%s_%s" % (var_name, i)] = val
            else:
                new_dict[var_name] = vals

        return new_dict


    def preprocess_data(self):
        for batch_dict in self.data_json:
            new_instances = []
            for instance_dict in batch_dict["instances"]:
                new_instances.append(self.unroll_instance_dict(instance_dict))
                
            batch_dict["instances"] = new_instances

    def get_batch(self, batch_name):
        for batch_dict in self.data_json:
            if batch_dict["batch_name"] == batch_name:
                return batch_dict["hypers"], batch_dict["instances"]
        assert False, "Unable to find data batch with name %s" % batch_name

    def get_batch_names(self):
        return [b["batch_name"] for b in self.data_json]

    def get_hypers_names(self):
        return set([b["hypers"] for b in self.data_json])

    def instance_list_to_batch(self, instance_dicts, perm):
        """
        Instances come in lists of one instance per element of the list.
        Each instance is a dictionary mapping variable names to values.
        Here, we reshape that list of instances into a dictionary that maps
        each variable name to a batched version (an array of values).
        """
        batched_vals = defaultdict(list)
        none_masks = defaultdict(list)
        for instance_dict in instance_dicts:
            for name, vals in instance_dict.iteritems():
                if vals is None:
                    batched_vals[name].append(0)
                    none_masks[name].append(0)
                else:
                    batched_vals[name].append(vals)
                    none_masks[name].append(1)
                #batched_vals[name].append(vals)
        
        value_mask_dict = {}
        
        for name, vals in batched_vals.iteritems():
            batched_vals[name] = np.array(vals)[perm]
            none_masks[name] = np.array(none_masks[name])[perm]

            value_mask_dict[name] = {
                "values": batched_vals[name],
                "masks"  : none_masks[name]
            }

        return value_mask_dict

    def get_random_minibatch(self, batch_name, batch_size):
        hypers_name, instance_list = self.get_batch(batch_name)
        num_instances = len(instance_list)
        perm = np.random.permutation(num_instances)
        if batch_size != -1 and batch_size < num_instances:
            perm = perm[:batch_size]

        minibatch = self.instance_list_to_batch(instance_list, perm)
        return hypers_name, minibatch
