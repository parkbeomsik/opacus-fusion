import torch

def is_equal(path_1, path_2, verbose=True):

    grad_1 = torch.load(path_1)
    grad_2 = torch.load(path_2)

    equal = True
    for k in grad_1:
        # print(k, grad_1[k].shape, grad_2[k].shape)
        # if not torch.equal(grad_1[k], grad_2[k]):
        if (torch.isclose(grad_1[k], grad_2[k]).sum().item()/grad_1[k].numel() < 0.98):
            if verbose:
                # print(torch.isclose(grad_1[k], grad_2[k]))
                print(f"===== {k}.grad_sample is not equal! =======")
                print(f"{torch.sum(torch.isclose(grad_1[k], grad_2[k])).item()/grad_1[k].numel()*100:.2f} % are equal")
                # print(f"<<< {path_1}")
                # print(grad_1[k].shape)
                # print(grad_1[k])
                # print(f">>> {path_2}")
                # print(grad_2[k].shape)
                # print(grad_2[k])

                # print((grad_1[k]/grad_2[k]).flatten()[0])

                torch.set_printoptions(profile="full")
                with open(f"value_test/grad_sample/{path_1.split('/')[-1]}_{k}_grad_sample", "w") as f:
                    f.write(grad_1[k].__str__())
                with open(f"value_test/grad_sample/{path_2.split('/')[-1]}_{k}_grad_sample", "w") as f:
                    f.write(grad_2[k].__str__())
                torch.set_printoptions(profile="default")

            equal = False
            # break
        else:
            if verbose:
                print(f"===== {k}.grad_sample s are equal! {torch.sum(torch.isclose(grad_1[k], grad_2[k])).item()/grad_1[k].numel()*100:.2f}% =======")
                # print(grad_1[k])
                # print(grad_2[k])

    if verbose:
        if equal:
            print("Two grads are equal! :)")
        else:
            print("Two grads are not equal :(")

    return equal

if __name__ == "__main__":
    is_equal("value_test/value_data/bert-base_grad_ref.pt",
             "value_test/value_data/bert-base_grad_elegant_hooks.pt", True)
            