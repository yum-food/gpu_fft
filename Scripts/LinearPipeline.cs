/*
    Udon# Linear Pipeline Controller

    This script manages a linear chain of image effects. It takes a source texture
    and processes it through a sequence of materials, storing the result of each
    step in a corresponding RenderTexture.

    Setup:
    1.  Create an empty GameObject in your scene.
    2.  Add an UdonBehaviour component to it.
    3.  Create this Udon# script in your project and assign it to the UdonBehaviour.
    4.  Create all the Materials and RenderTextures you need for your pipeline.
    5.  In the Inspector, assign the Source Input, Materials, and Output RenderTextures.
        - The size of the Effect Materials and Pipeline Outputs arrays MUST be the same.
        - The order of materials and textures in the arrays determines the pipeline order.
    6.  Choose your desired execution mode (Run On Start, Run Continuously).
*/

using UdonSharp;
using UnityEngine;
using VRC.SDKBase;

[UdonBehaviourSyncMode(BehaviourSyncMode.None)]
public class LinearPipeline : UdonSharpBehaviour
{
    [Header("Pipeline Assets")]
    [Tooltip("The initial texture to start the pipeline with.")]
    public Texture sourceInput;

    [Tooltip("The materials to apply in sequence. The order matters.")]
    public Material[] effectMaterials;

    [Tooltip("The RenderTextures to store the output of each step. MUST be the same size as the materials array.")]
    public RenderTexture[] pipelineOutputs;

    [Header("Execution Mode")]
    [Tooltip("If true, the pipeline will run once when the world loads.")]
    public bool runOnStart = true;

    [Tooltip("If true, the pipeline will run every frame. Use with caution, can be performance-intensive.")]
    public bool runContinuously = false;

    void Start()
    {
        if (runOnStart)
        {
            _RunPipeline();
        }
    }

    void Update()
    {
        if (runContinuously)
        {
            _RunPipeline();
        }
    }

    /// <summary>
    /// This public method can be called by other Udon scripts or UI events to run the pipeline.
    /// </summary>
    public void _RunPipeline()
    {
        // --- Pre-flight Checks ---
        if (sourceInput == null)
        {
            Debug.LogError("[LinearPipeline] Source Input is not assigned!", this);
            return;
        }

        if (effectMaterials == null || effectMaterials.Length == 0)
        {
            Debug.LogError("[LinearPipeline] No Effect Materials have been assigned!", this);
            return;
        }

        if (pipelineOutputs == null || pipelineOutputs.Length == 0)
        {
            Debug.LogError("[LinearPipeline] No Pipeline Outputs have been assigned!", this);
            return;
        }

        if (effectMaterials.Length != pipelineOutputs.Length)
        {
            Debug.LogError("[LinearPipeline] The number of materials does not match the number of output textures!", this);
            return;
        }

        // --- Run Pipeline ---

        // 1. First Blit: From the main source to the first texture in our chain.
        VRCGraphics.Blit(sourceInput, pipelineOutputs[0], effectMaterials[0], -1);

        // 2. Loop through the rest of the chain.
        for (int i = 1; i < effectMaterials.Length; i++)
        {
            // The source for this step is the output from the previous step.
            Texture sourceForThisStep = pipelineOutputs[i - 1];

            // The destination is the current output texture.
            RenderTexture destForThisStep = pipelineOutputs[i];

            // The material for this step.
            Material materialForThisStep = effectMaterials[i];

            VRCGraphics.Blit(sourceForThisStep, destForThisStep, materialForThisStep, -1);
        }
    }
}

