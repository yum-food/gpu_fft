using UdonSharp;
using UnityEngine;
using VRC.SDKBase;
using VRC.Udon;

public class PipelineExecutor : UdonSharpBehaviour
{
    public LinearPipeline[] pipelines;

    void Update()
    {
        if (pipelines == null) return;

        for (int i = 0; i < pipelines.Length; i++)
        {
            if (pipelines[i] != null)
            {
                pipelines[i].RunPipeline();
            }
        }
    }
}

