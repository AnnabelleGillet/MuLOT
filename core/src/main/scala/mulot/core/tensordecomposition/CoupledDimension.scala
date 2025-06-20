package mulot.core.tensordecomposition

import mulot.core.Tensor

/**
 * Indicates a coupling between two tensors.
 *
 * @param tensor1 the first tensor.
 * @param tensor2 the second tensor.
 * @param mapping the coupled dimensions, with the key being the index of the dimension in tensor1,
 *                and the value being the index of the dimension in tensor2.
 */
case class CoupledDimension[T <: Tensor](tensor1: T, tensor2: T, mapping: Map[Int, Int])
