# Etapa 19 — Cloud & Infra de GPU
# Cluster de inferência na AWS: node group com GPU dedicado para
# serving do modelo, separado do node group da API (que é CPU-only).

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_ecr_repository" "llm_support_api" {
  name                 = "llm-support-api"
  image_tag_mutability = "IMMUTABLE"
}

# Node group de GPU dedicado ao serving do modelo (custo por hora alto —
# usar autoscaling agressivo com min=0 fora de horário de pico, se possível)
resource "aws_eks_node_group" "gpu_inference" {
  cluster_name    = var.eks_cluster_name
  node_group_name = "gpu-inference"
  node_role_arn   = var.node_role_arn
  subnet_ids      = var.private_subnet_ids

  instance_types = ["g5.2xlarge"] # 1x A10G — bom custo-benefício para 7B-13B

  scaling_config {
    desired_size = 2
    min_size     = 1
    max_size     = 6
  }

  labels = {
    "workload" = "llm-inference"
  }

  taint {
    key    = "nvidia.com/gpu"
    value  = "true"
    effect = "NO_SCHEDULE"
  }
}

resource "aws_s3_bucket" "kb_artifacts" {
  bucket = "${var.project_name}-knowledge-base"
}

variable "aws_region" { default = "us-east-1" }
variable "project_name" { default = "llmops-journey" }
variable "eks_cluster_name" { type = string }
variable "node_role_arn" { type = string }
variable "private_subnet_ids" { type = list(string) }
