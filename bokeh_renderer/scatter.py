#!/user/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import cupy
import re

kernel_Render_updateOutput = '''

    extern "C" __global__ void kernel_Render_updateOutput(
        const int n,
        const float* image,          // original image
        const float* defocus,        // signed defocus map
        float* bokehCum,             // cumulative bokeh image
        float* weightCum             // cumulative weight map
    )
    {
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intN = ( intIndex / SIZE_3(weightCum) / SIZE_2(weightCum) / SIZE_1(weightCum) ) % SIZE_0(weightCum);
            // const int intC = ( intIndex / SIZE_3(weightCum) / SIZE_2(weightCum)                     ) % SIZE_1(weightCum);
            const int intY = ( intIndex / SIZE_3(weightCum)                                         ) % SIZE_2(weightCum);
            const int intX = ( intIndex                                                             ) % SIZE_3(weightCum);


            float fltDefocus = VALUE_4(defocus, intN, 0, intY, intX);
            float fltRadius = fabsf(fltDefocus);


            for (int intDeltaY = -(int)(fltRadius)-1; intDeltaY <= (int)(fltRadius)+1; ++intDeltaY) {
                for (int intDeltaX = -(int)(fltRadius)-1; intDeltaX <= (int)(fltRadius)+1; ++intDeltaX) {

                    int intNeighborY = intY + intDeltaY;
                    int intNeighborX = intX + intDeltaX;

                    if ((intNeighborY >= 0) && (intNeighborY < SIZE_2(weightCum)) && (intNeighborX >= 0) && (intNeighborX < SIZE_3(weightCum))) {
                        float fltDist = sqrtf((float)(intDeltaY)*(float)(intDeltaY) + (float)(intDeltaX)*(float)(intDeltaX));
                        float fltWeight = (0.5 + 0.5 * tanhf(4 * (fltRadius - fltDist))) / (fltRadius * fltRadius + 0.2);
                        atomicAdd(&weightCum[OFFSET_4(weightCum, intN, 0, intNeighborY, intNeighborX)], fltWeight);
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 0, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 0, intY, intX));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 1, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 1, intY, intX));
                        atomicAdd(&bokehCum[OFFSET_4(bokehCum, intN, 2, intNeighborY, intNeighborX)], fltWeight * VALUE_4(image, intN, 2, intY, intX));
                    }
                }
            }
        }
    }

'''


kernel_Render_updateGradInput = '''

    extern "C" __global__ void kernel_Render_updateGradInput(
        const int n,
        const float* image,         // original image
        const float* defocus,       // signed defocus map
        const float* gradBokehCum,  // gradient of cumulative bokeh image
        const float* gradWeightCum, // gradient of cumulative weight map
        float* gradImage,           // gradient of original image
        float* gradDefocus         // gradient of signed defocus map
    )
    {
        for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
            const int intN = ( intIndex / SIZE_3(gradDefocus) / SIZE_2(gradDefocus) / SIZE_1(gradDefocus) ) % SIZE_0(gradDefocus);
            // const int intC = ( intIndex / SIZE_3(gradDefocus) / SIZE_2(gradDefocus)                       ) % SIZE_1(gradDefocus);
            const int intY = ( intIndex / SIZE_3(gradDefocus)                                             ) % SIZE_2(gradDefocus);
            const int intX = ( intIndex                                                                   ) % SIZE_3(gradDefocus);


            float fltDefocus = VALUE_4(defocus, intN, 0, intY, intX);
            float fltRadius = fabsf(fltDefocus);
            float fltRadiusSquare = fltRadius * fltRadius + 0.2;
            float dRadius_div_dDefocus = 1.0;

            if (fltDefocus < 0) {
                dRadius_div_dDefocus = -1.0;
            }

            float fltGradImageR = 0.0;
            float fltGradImageG = 0.0;
            float fltGradImageB = 0.0;
            float fltGradDefocus = 0.0;


            for (int intDeltaY = -(int)(fltRadius)-1; intDeltaY <= (int)(fltRadius)+1; ++intDeltaY) {
                for (int intDeltaX = -(int)(fltRadius)-1; intDeltaX <= (int)(fltRadius)+1; ++intDeltaX) {

                    int intNeighborY = intY + intDeltaY;
                    int intNeighborX = intX + intDeltaX;

                    if ((intNeighborY >= 0) & (intNeighborY < SIZE_2(gradDefocus)) & (intNeighborX >= 0) & (intNeighborX < SIZE_3(gradDefocus))) {
                        float fltDist = sqrtf((float)(intDeltaY)*(float)(intDeltaY) + (float)(intDeltaX)*(float)(intDeltaX));
                        float fltTanh = tanhf(4 * (fltRadius - fltDist));
                        float dWeight_div_dDefocus = dRadius_div_dDefocus * (2 * (1 - fltTanh * fltTanh) / fltRadiusSquare - (1 + fltTanh) * fltRadius / fltRadiusSquare / fltRadiusSquare);
                        float dWeight_div_dMask = (0.5 + 0.5 * fltTanh) / fltRadiusSquare;
                        float fltWeight = dWeight_div_dMask;
                        
                        fltGradImageR += VALUE_4(gradBokehCum, intN, 0, intNeighborY, intNeighborX) * fltWeight;
                        fltGradImageG += VALUE_4(gradBokehCum, intN, 1, intNeighborY, intNeighborX) * fltWeight;
                        fltGradImageB += VALUE_4(gradBokehCum, intN, 2, intNeighborY, intNeighborX) * fltWeight;
                        
                        fltGradDefocus += VALUE_4(gradBokehCum, intN, 0, intNeighborY, intNeighborX) * VALUE_4(image, intN, 0, intY, intX) * dWeight_div_dDefocus;
                        fltGradDefocus += VALUE_4(gradBokehCum, intN, 1, intNeighborY, intNeighborX) * VALUE_4(image, intN, 1, intY, intX) * dWeight_div_dDefocus;
                        fltGradDefocus += VALUE_4(gradBokehCum, intN, 2, intNeighborY, intNeighborX) * VALUE_4(image, intN, 2, intY, intX) * dWeight_div_dDefocus;
                        fltGradDefocus += VALUE_4(gradWeightCum, intN, 0, intNeighborY, intNeighborX) * dWeight_div_dDefocus;
                    
                    }
                }
            }

            gradImage[OFFSET_4(gradImage, intN, 0, intY, intX)] = fltGradImageR;
            gradImage[OFFSET_4(gradImage, intN, 1, intY, intX)] = fltGradImageG;
            gradImage[OFFSET_4(gradImage, intN, 2, intY, intX)] = fltGradImageB;
            
            gradDefocus[OFFSET_4(gradDefocus, intN, 0, intY, intX)] = fltGradDefocus;

        }
    }

'''






def cupy_kernel(strFunction, objVariables):
    strKernel = globals()[strFunction]

    while True:
        objMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArg = int(objMatch.group(2))

        strTensor = objMatch.group(4)
        intSizes = objVariables[strTensor].size()

        strKernel = strKernel.replace(objMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), '(' + str.join('+', strIndex) + ')')
    # end

    while True:
        objMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objMatch is None:
            break
        # end

        intArgs = int(objMatch.group(2))
        strArgs = objMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(
            intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end


# @cupy.util.memoize(for_each_device=True)
@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end


class _FunctionRender(torch.autograd.Function):
    @staticmethod
    # def forward(self, image, defocus, mask):
    def forward(self, image, defocus):
        # self.save_for_backward(image, defocus, mask)
        self.save_for_backward(image, defocus)

        bokeh_cum = torch.zeros_like(image)
        weight_cum = torch.zeros_like(defocus)

        if defocus.is_cuda is True:
            n = weight_cum.nelement()
            cupy_launch('kernel_Render_updateOutput', cupy_kernel('kernel_Render_updateOutput', {
                'image': image,
                'defocus': defocus,
                'bokehCum': bokeh_cum,
                'weightCum': weight_cum
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[
                    cupy.int(n),
                    image.data_ptr(),
                    defocus.data_ptr(),
                    bokeh_cum.data_ptr(),
                    weight_cum.data_ptr()
                ]
            )
        # end

        elif defocus.is_cuda is False:
            raise NotImplementedError()
        # end

        return bokeh_cum, weight_cum
    # end

    @staticmethod
    def backward(self, grad_bokeh_cum, grad_weight_cum):
        # image, defocus, mask = self.saved_tensors
        image, defocus = self.saved_tensors

        grad_image = torch.zeros_like(image) if self.needs_input_grad[1] is True else None
        grad_defocus = torch.zeros_like(defocus) if self.needs_input_grad[1] is True else None
        # grad_mask = torch.zeros_like(mask) if self.needs_input_grad[2] is True else None

        if defocus.is_cuda is True:
            if grad_defocus is not None:
                n = grad_defocus.nelement()
                cupy_launch('kernel_Render_updateGradInput', cupy_kernel('kernel_Render_updateGradInput', {
                    'image': image,
                    'defocus': defocus,
                    'gradBokehCum': grad_bokeh_cum,
                    'gradWeightCum': grad_weight_cum,
                    'gradImage': grad_image,
                    'gradDefocus': grad_defocus
                }))(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[
                        cupy.int(n),
                        image.data_ptr(),
                        defocus.data_ptr(),
                        grad_bokeh_cum.data_ptr(),
                        grad_weight_cum.data_ptr(),
                        grad_image.data_ptr(),
                        grad_defocus.data_ptr()
                    ]
                )
            # end

        elif defocus.is_cuda is False:
            raise NotImplementedError()
        #end

        return grad_image, grad_defocus 
# end


# def FunctionRender(image, defocus, mask):
def FunctionRender(image, defocus):
    # bokeh_cum, weight_cum = _FunctionRender.apply(image, defocus, mask)
    bokeh_cum, weight_cum = _FunctionRender.apply(image, defocus)
    return bokeh_cum, weight_cum
# end

class ModuleRenderScatter(torch.nn.Module):
    """Rendering Circular Circle-of-Confusion
    """
    def __init__(self):
        super(ModuleRenderScatter, self).__init__()
    # end

    def forward(self, image, defocus):
        bokeh_cum, weight_cum = FunctionRender(image, defocus)
        bokeh = bokeh_cum / weight_cum
        return bokeh
    # end
# end