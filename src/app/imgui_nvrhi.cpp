/*
* Copyright (c) 2014-2025, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/

/*
License for Dear ImGui

Copyright (c) 2014-2025 Omar Cornut

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stddef.h>

#include <imgui.h>

#include <nvrhi/nvrhi.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/app/imgui_nvrhi.h>
#include <donut/core/log.h>

#if DONUT_WITH_STATIC_SHADERS
#if DONUT_WITH_DX11
#include "compiled_shaders/imgui_vertex.dxbc.h"
#include "compiled_shaders/imgui_pixel.dxbc.h"
#endif
#if DONUT_WITH_DX12
#include "compiled_shaders/imgui_vertex.dxil.h"
#include "compiled_shaders/imgui_pixel.dxil.h"
#endif
#if DONUT_WITH_VULKAN
#include "compiled_shaders/imgui_vertex.spirv.h"
#include "compiled_shaders/imgui_pixel.spirv.h"
#endif
#endif

using namespace donut::engine;
using namespace donut::app;

struct VERTEX_CONSTANT_BUFFER
{
    float        mvp[4][4];
};

// Dear ImGui 1.92+ dynamic textures. ImGui owns the pixel data and tells us,
// per frame and per texture, what it needs done; we own the GPU resource. This
// is what lets the font atlas rasterize glyphs on demand instead of pre-baking
// a fixed GlyphRanges into one static texture at startup.
//
// nvrhi has no sub-rectangle texture write (ICommandList::writeTexture takes a
// whole mip), so a WantUpdates re-uploads the full image rather than the dirty
// rect ImGui hands us. The atlas is small and grows rarely, so this is a few
// hundred KB on the frames where it happens - cheaper than the alternative it
// replaces, which was holding every glyph resident forever.
void ImGui_NVRHI::updateTexture(ImTextureData* tex)
{
    if (tex->Status == ImTextureStatus_WantCreate)
    {
        nvrhi::TextureDesc textureDesc;
        textureDesc.width = uint32_t(tex->Width);
        textureDesc.height = uint32_t(tex->Height);
        textureDesc.format = (tex->Format == ImTextureFormat_Alpha8)
                           ? nvrhi::Format::R8_UNORM : nvrhi::Format::RGBA8_UNORM;
        textureDesc.debugName = "ImGui texture";
        // keepInitialState, NOT setPermanentTextureState: these are written
        // again every time a new glyph is rasterized into them, and a
        // permanent state is immutable in nvrhi - the first update failed
        // validation with "Permanent texture ... doesn't have the right state
        // bits. Required: 0x800 (CopyDest), present: 0x20 (ShaderResource)".
        // The legacy font texture below gets away with permanent state only
        // because it is uploaded exactly once. keepInitialState also carries
        // the state ACROSS command lists, which plain tracking does not - the
        // next thing this hit was "Unknown prior state of texture", once per
        // frame, because every open() starts tracking from scratch.
        textureDesc.initialState = nvrhi::ResourceStates::ShaderResource;
        textureDesc.keepInitialState = true;

        nvrhi::TextureHandle texture = m_device->createTexture(textureDesc);
        if (texture == nullptr)
            return;   // leave Status alone; ImGui will ask again next frame

        // writeTexture transitions to CopyDest and keepInitialState restores
        // ShaderResource at close, so the draws that sample it - in a
        // different command list - need nothing further.
        m_commandList->open();
        m_commandList->writeTexture(texture, 0, 0, tex->GetPixels(), size_t(tex->GetPitch()));
        m_commandList->close();
        m_device->executeCommandList(m_commandList);

        tex->SetTexID(ImTextureID(texture.Get()));
        tex->SetStatus(ImTextureStatus_OK);
        managedTextures[tex] = texture;

        // Rare (the atlas grows by doubling and then stops), and the one line
        // that says the dynamic path is actually live - and how big the atlas
        // grew to, which is the whole point of it being dynamic.
        donut::log::info("ImGui texture #%d created: %dx%d, %.2f MB (%d live)",
                         tex->UniqueID, tex->Width, tex->Height,
                         double(tex->GetSizeInBytes()) / (1024.0 * 1024.0), int(managedTextures.size()));
    }
    else if (tex->Status == ImTextureStatus_WantUpdates)
    {
        auto it = managedTextures.find(tex);
        if (it == managedTextures.end())
            return;

        m_commandList->open();
        m_commandList->writeTexture(it->second, 0, 0, tex->GetPixels(), size_t(tex->GetPitch()));
        m_commandList->close();
        m_device->executeCommandList(m_commandList);

        tex->SetStatus(ImTextureStatus_OK);
    }
    else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames > 0)
    {
        destroyTexture(tex);
    }
}

void ImGui_NVRHI::destroyTexture(ImTextureData* tex)
{
    auto it = managedTextures.find(tex);
    if (it != managedTextures.end())
    {
        // The binding set caches this texture; it must go with it or the next
        // texture allocated at the same address inherits a stale binding.
        bindingsCache.erase(it->second.Get());
        managedTextures.erase(it);
        donut::log::info("ImGui texture #%d destroyed (%d live)", tex->UniqueID, int(managedTextures.size()));
    }
    tex->SetTexID(ImTextureID_Invalid);
    tex->SetStatus(ImTextureStatus_Destroyed);
}

void ImGui_NVRHI::updateTextures(ImDrawData* drawData)
{
    if (drawData == nullptr || drawData->Textures == nullptr)
        return;
    for (ImTextureData* tex : *drawData->Textures)
        if (tex != nullptr && tex->Status != ImTextureStatus_OK)
            updateTexture(tex);
}

bool ImGui_NVRHI::updateFontTexture()
{
    ImGuiIO& io = ImGui::GetIO();

    // Dynamic textures: the atlas creates and grows itself through
    // updateTextures() during render, and there is deliberately nothing to
    // pre-build here. Touching io.Fonts->TexRef in this mode would detach the
    // atlas from the ImTextureData it manages.
    if (io.BackendFlags & ImGuiBackendFlags_RendererHasTextures)
        return true;

    // If the font texture exists and is bound to ImGui, we're done.
    // Note: ImGui_Renderer will reset io.Fonts->TexRef when new fonts are added.
    if (fontTexture && io.Fonts->TexRef.GetTexID())
        return true;

    unsigned char *pixels;
    int width, height;

    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    if (!pixels)
        return false;

    nvrhi::TextureDesc textureDesc;
    textureDesc.width = width;
    textureDesc.height = height;
    textureDesc.format = nvrhi::Format::RGBA8_UNORM;
    textureDesc.debugName = "ImGui font texture";

    fontTexture = m_device->createTexture(textureDesc);

    if (fontTexture == nullptr)
        return false;

    m_commandList->open();

    m_commandList->beginTrackingTextureState(fontTexture, nvrhi::AllSubresources, nvrhi::ResourceStates::Common);

    m_commandList->writeTexture(fontTexture, 0, 0, pixels, width * 4);

    m_commandList->setPermanentTextureState(fontTexture, nvrhi::ResourceStates::ShaderResource);
    m_commandList->commitBarriers();

    m_commandList->close();
    m_device->executeCommandList(m_commandList);

    io.Fonts->TexRef = ImTextureRef(fontTexture.Get());

    return true;
}

bool ImGui_NVRHI::init(nvrhi::IDevice* device, std::shared_ptr<ShaderFactory> shaderFactory)
{
    m_device = device;

    // See updateTexture(): opt into the dynamic font atlas. Must be set before
    // the first atlas build, since ImFontAtlasBuildMain latches it to decide
    // between on-demand rasterization and legacy pre-baking.
    ImGui::GetIO().BackendFlags |= ImGuiBackendFlags_RendererHasTextures;

    m_commandList = m_device->createCommandList();

    vertexShader = shaderFactory->CreateAutoShader("donut/imgui_vertex", "main", DONUT_MAKE_PLATFORM_SHADER(g_imgui_vertex), nullptr, nvrhi::ShaderType::Vertex);
    pixelShader = shaderFactory->CreateAutoShader("donut/imgui_pixel", "main", DONUT_MAKE_PLATFORM_SHADER(g_imgui_pixel), nullptr, nvrhi::ShaderType::Pixel);
    
    if (!vertexShader || !pixelShader)
    {
        log::error("Failed to create an ImGUI shader");
        return false;
    } 

    // create attribute layout object
    nvrhi::VertexAttributeDesc vertexAttribLayout[] = {
        { "POSITION", nvrhi::Format::RG32_FLOAT,  1, 0, offsetof(ImDrawVert,pos), sizeof(ImDrawVert), false },
        { "TEXCOORD", nvrhi::Format::RG32_FLOAT,  1, 0, offsetof(ImDrawVert,uv),  sizeof(ImDrawVert), false },
        { "COLOR",    nvrhi::Format::RGBA8_UNORM, 1, 0, offsetof(ImDrawVert,col), sizeof(ImDrawVert), false },
    };

    shaderAttribLayout = m_device->createInputLayout(vertexAttribLayout, sizeof(vertexAttribLayout) / sizeof(vertexAttribLayout[0]), vertexShader);

    // create PSO
    {
        nvrhi::BlendState blendState;
        blendState.targets[0].setBlendEnable(true)
            .setSrcBlend(nvrhi::BlendFactor::SrcAlpha)
            .setDestBlend(nvrhi::BlendFactor::InvSrcAlpha)
            // Accumulate COVERAGE in alpha: dstA = srcA + dstA*(1-srcA).
            //
            // Upstream used SrcBlendAlpha=InvSrcAlpha, DestBlendAlpha=Zero,
            // which yields dstA = srcA*(1-srcA) - zero for both a fully opaque
            // and a fully transparent pixel, and never above 0.25. Harmless
            // when the target is the swapchain, where alpha is never read, but
            // it makes the render target unusable as a source: the most solid
            // parts of the UI carry the LOWEST alpha.
            //
            // dxr_native composites the UI from an offscreen target onto a
            // world-space quad in VR, so it needs alpha to mean coverage. RGB
            // blending is unchanged, so the result is PREMULTIPLIED - composite
            // with `ui.rgb + dst*(1-ui.a)`, not a lerp.
            .setSrcBlendAlpha(nvrhi::BlendFactor::One)
            .setDestBlendAlpha(nvrhi::BlendFactor::InvSrcAlpha);

        auto rasterState = nvrhi::RasterState()
            .setFillSolid()
            .setCullNone()
            .setScissorEnable(true)
            .setDepthClipEnable(true);

        auto depthStencilState = nvrhi::DepthStencilState()
            .disableDepthTest()
            .enableDepthWrite()
            .disableStencil()
            .setDepthFunc(nvrhi::ComparisonFunc::Always);

        nvrhi::RenderState renderState;
        renderState.blendState = blendState;
        renderState.depthStencilState = depthStencilState;
        renderState.rasterState = rasterState;

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc.visibility = nvrhi::ShaderType::All;
        layoutDesc.bindings = { 
            nvrhi::BindingLayoutItem::PushConstants(0, sizeof(float) * 2),
            nvrhi::BindingLayoutItem::Texture_SRV(0),
            nvrhi::BindingLayoutItem::Sampler(0) 
        };
        bindingLayout = m_device->createBindingLayout(layoutDesc);

        basePSODesc.primType = nvrhi::PrimitiveType::TriangleList;
        basePSODesc.inputLayout = shaderAttribLayout;
        basePSODesc.VS = vertexShader;
        basePSODesc.PS = pixelShader;
        basePSODesc.renderState = renderState;
        basePSODesc.bindingLayouts = { bindingLayout };
    }

    {
        const auto desc = nvrhi::SamplerDesc()
            .setAllAddressModes(nvrhi::SamplerAddressMode::Wrap)
            .setAllFilters(true);

        fontSampler = m_device->createSampler(desc);

        if (fontSampler == nullptr)
            return false;
    }

    return true;
}

bool ImGui_NVRHI::reallocateBuffer(nvrhi::BufferHandle& buffer, size_t requiredSize, size_t reallocateSize, const bool indexBuffer)
{
    if (buffer == nullptr || size_t(buffer->getDesc().byteSize) < requiredSize)
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = uint32_t(reallocateSize);
        desc.structStride = 0;
        desc.debugName = indexBuffer ? "ImGui index buffer" : "ImGui vertex buffer";
        desc.canHaveUAVs = false;
        desc.isVertexBuffer = !indexBuffer;
        desc.isIndexBuffer = indexBuffer;
        desc.isDrawIndirectArgs = false;
        desc.isVolatile = false;
        desc.initialState = indexBuffer ? nvrhi::ResourceStates::IndexBuffer : nvrhi::ResourceStates::VertexBuffer;
        desc.keepInitialState = true;

        buffer = m_device->createBuffer(desc);

        if (!buffer)
        {
            return false;
        }
    }

    return true;
}

nvrhi::IGraphicsPipeline* ImGui_NVRHI::getPSO(nvrhi::FramebufferInfo const& framebufferInfo)
{
    if (pso)
        return pso;

    pso = m_device->createGraphicsPipeline(basePSODesc, framebufferInfo);
    assert(pso);

    return pso;
}

nvrhi::IBindingSet* ImGui_NVRHI::getBindingSet(nvrhi::ITexture* texture)
{
    auto iter = bindingsCache.find(texture);
    if (iter != bindingsCache.end())
    {
        return iter->second;
    }

    nvrhi::BindingSetDesc desc;

    desc.bindings = {
        nvrhi::BindingSetItem::PushConstants(0, sizeof(float) * 2),
        nvrhi::BindingSetItem::Texture_SRV(0, texture),
        nvrhi::BindingSetItem::Sampler(0, fontSampler)
    };

    nvrhi::BindingSetHandle binding;
    binding = m_device->createBindingSet(desc, bindingLayout);
    assert(binding);

    bindingsCache[texture] = binding;
    return binding;
}

bool ImGui_NVRHI::updateGeometry(nvrhi::ICommandList* commandList)
{
    ImDrawData *drawData = ImGui::GetDrawData();

    // create/resize vertex and index buffers if needed
    if (!reallocateBuffer(vertexBuffer, 
        drawData->TotalVtxCount * sizeof(ImDrawVert), 
        (drawData->TotalVtxCount + 5000) * sizeof(ImDrawVert), 
        false))
    {
        return false;
    }

    if (!reallocateBuffer(indexBuffer,
        drawData->TotalIdxCount * sizeof(ImDrawIdx),
        (drawData->TotalIdxCount + 5000) * sizeof(ImDrawIdx),
        true))
    {
        return false;
    }

    vtxBuffer.resize(vertexBuffer->getDesc().byteSize / sizeof(ImDrawVert));
    idxBuffer.resize(indexBuffer->getDesc().byteSize / sizeof(ImDrawIdx));

    // copy and convert all vertices into a single contiguous buffer
    ImDrawVert *vtxDst = &vtxBuffer[0];
    ImDrawIdx *idxDst = &idxBuffer[0];

    for(int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList *cmdList = drawData->CmdLists[n];

        memcpy(vtxDst, cmdList->VtxBuffer.Data, cmdList->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idxDst, cmdList->IdxBuffer.Data, cmdList->IdxBuffer.Size * sizeof(ImDrawIdx));

        vtxDst += cmdList->VtxBuffer.Size;
        idxDst += cmdList->IdxBuffer.Size;
    }
    
    commandList->writeBuffer(vertexBuffer, &vtxBuffer[0], vertexBuffer->getDesc().byteSize);
    commandList->writeBuffer(indexBuffer, &idxBuffer[0], indexBuffer->getDesc().byteSize);

    return true;
}

bool ImGui_NVRHI::render(nvrhi::IFramebuffer* framebuffer)
{
    ImDrawData *drawData = ImGui::GetDrawData();
    const auto& io = ImGui::GetIO();

    // Before opening the render command list: these submit their own uploads,
    // and a newly grown atlas has to be resident before the draws that sample
    // it. ImGui asks for this every frame; almost always there is nothing to do.
    updateTextures(drawData);

    m_commandList->open();
    m_commandList->beginMarker("ImGUI");

    if (!updateGeometry(m_commandList))
    {
        m_commandList->close();
        return false;
    }

    // handle DPI scaling
    drawData->ScaleClipRects(io.DisplayFramebufferScale);

    float invDisplaySize[2] = { 1.f / io.DisplaySize.x, 1.f / io.DisplaySize.y };

    // set up graphics state
    nvrhi::GraphicsState drawState;

    drawState.framebuffer = framebuffer;
    assert(drawState.framebuffer);
    
    drawState.pipeline = getPSO(framebuffer->getFramebufferInfo());

    drawState.viewport.viewports.push_back(nvrhi::Viewport(io.DisplaySize.x * io.DisplayFramebufferScale.x,
                                           io.DisplaySize.y * io.DisplayFramebufferScale.y));
    drawState.viewport.scissorRects.resize(1);  // updated below

    nvrhi::VertexBufferBinding vbufBinding;
    vbufBinding.buffer = vertexBuffer;
    vbufBinding.slot = 0;
    vbufBinding.offset = 0;
    drawState.vertexBuffers.push_back(vbufBinding);

    drawState.indexBuffer.buffer = indexBuffer;
    drawState.indexBuffer.format = (sizeof(ImDrawIdx) == 2 ? nvrhi::Format::R16_UINT : nvrhi::Format::R32_UINT);
    drawState.indexBuffer.offset = 0;

    // render command lists
    int vtxOffset = 0;
    int idxOffset = 0;
    for(int n = 0; n < drawData->CmdListsCount; n++)
    {
        const ImDrawList *cmdList = drawData->CmdLists[n];
        for(int i = 0; i < cmdList->CmdBuffer.Size; i++)
        {
            const ImDrawCmd *pCmd = &cmdList->CmdBuffer[i];

            if (pCmd->UserCallback)
            {
                pCmd->UserCallback(cmdList, pCmd);
            } else {
                drawState.bindings = { getBindingSet((nvrhi::ITexture*)pCmd->TexRef.GetTexID()) };
                assert(drawState.bindings[0]);

                drawState.viewport.scissorRects[0] = nvrhi::Rect(int(pCmd->ClipRect.x),
                                                                 int(pCmd->ClipRect.z),
                                                                 int(pCmd->ClipRect.y),
                                                                 int(pCmd->ClipRect.w));

                nvrhi::DrawArguments drawArguments;
                drawArguments.vertexCount = pCmd->ElemCount;
                drawArguments.startIndexLocation = idxOffset;
                drawArguments.startVertexLocation = vtxOffset;

                m_commandList->setGraphicsState(drawState);
                m_commandList->setPushConstants(invDisplaySize, sizeof(invDisplaySize));
                m_commandList->drawIndexed(drawArguments);
            }

            idxOffset += pCmd->ElemCount;
        }

        vtxOffset += cmdList->VtxBuffer.Size;
    }

    m_commandList->endMarker();
    m_commandList->close();
    m_device->executeCommandList(m_commandList);

    return true;
}

void ImGui_NVRHI::backbufferResizing()
{
    pso = nullptr;
}
