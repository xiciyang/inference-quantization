/**
 * 后端聚合入口：import 本模块即完成所有内置量化方法的注册。
 * 新增方法 = 在这里加一行 import + 新增一个方法模块，**核心逻辑零改动**。
 */
import './llmcompressor.mjs'

export * from './registry.mjs'
