package com.example.rulesearch

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform