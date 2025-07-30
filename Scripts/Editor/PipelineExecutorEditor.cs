#if UNITY_EDITOR

using UnityEngine;
using UnityEditor;
using System.Linq;

[CustomEditor(typeof(PipelineExecutor))]
public class PipelineExecutorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        PipelineExecutor executor = (PipelineExecutor)target;

        EditorGUILayout.Space();

        if (GUILayout.Button("Auto-Add All Pipelines"))
        {
            AutoAddPipelines(executor);
        }
    }

    private void AutoAddPipelines(PipelineExecutor executor)
    {
        // Find all LinearPipeline components in the current scene
        LinearPipeline[] allPipelines = Object.FindObjectsOfType<LinearPipeline>();

        // Sort by hierarchy order
        System.Array.Sort(allPipelines, (a, b) =>
        {
            // Get hierarchy paths and compare
            string pathA = GetHierarchyPath(a.transform);
            string pathB = GetHierarchyPath(b.transform);
            return pathA.CompareTo(pathB);
        });

        LinearPipeline[] existingPipelines = executor.pipelines ?? new LinearPipeline[0];
        LinearPipeline[] newPipelines = allPipelines.Where(p => p != null && !existingPipelines.Contains(p)).ToArray();

        if (newPipelines.Length > 0)
        {
            Undo.RecordObject(executor, "Auto-Add Pipelines");
            executor.pipelines = existingPipelines.Concat(newPipelines).ToArray();
            EditorUtility.SetDirty(executor);
            Debug.Log($"[PipelineExecutor] Added {newPipelines.Length} new pipelines. Total: {executor.pipelines.Length}");
        }
        else
        {
            Debug.Log("[PipelineExecutor] No new pipelines to add.");
        }
    }

    private string GetHierarchyPath(Transform transform)
    {
        System.Collections.Generic.List<int> indices = new System.Collections.Generic.List<int>();
        Transform current = transform;

        while (current != null)
        {
            indices.Add(current.GetSiblingIndex());
            current = current.parent;
        }

        indices.Reverse();
        return string.Join(".", indices.Select(i => i.ToString("D4")));
    }
}

#endif
